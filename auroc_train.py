import os
import sys
import warnings
import time
from torch.nn import modules
import yaml
from types import SimpleNamespace

import numpy as np
from numpy.random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from PIL import Image
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

from utils.eval import eval_net, eval_supcon
from utils.print_log import train_log
from models.resnet3d import SupConResNet3D, resnet3d
from utils.datasets import AortaDataset3D, LabelSampler, AortaDataset3DCenter
from models.SupCon import *
from models.losses import SupConLoss
from utils import transforms as MT

warnings.filterwarnings("ignore")
np.random.seed(63910)
torch.manual_seed(53152)
torch.cuda.manual_seed_all(7987)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # faster convolutions, but more memory



"""************************************************** Cross Entropy **************************************************"""
def create_net(device,
               model_name='resnet',
               model_depth=34,
               n_channels=1,
               n_classes=4,
               load_model=False,
               load_encoder=False,
               flag_3d=False,
               **kwargs):
    
    args = SimpleNamespace(**kwargs)

    if flag_3d:
        net = resnet3d(model_depth, n_channels=n_channels, n_classes=n_classes, conv1_t_size=3)
    else:
        if model_name.lower() == 'resnet':
            net = resnet(model_depth, n_channels=n_channels, n_classes=n_classes, entire=args.entire)
        elif model_name.lower() == 'efficientnet':
            net = models.efficientnet_b3(num_channels=n_channels, num_classes=n_classes)
            net.n_channels, net.n_classes, net.net_name = n_channels, n_classes, "EfficientNet-B3"

    train_log.info('**********************************************************************\n'
                 f'Network: {net.net_name}\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n'
                 f'\t3D model: {flag_3d}\n')

    if load_model:
        net.load_state_dict(torch.load(load_model, map_location=device))
        train_log.info(f'Model loaded from {load_model}')
    elif load_encoder:
        sup = SupConResNet(n_channels=n_channels, name=f'resnet{model_depth}', head='mlp', feat_dim=128)
        sup.load_state_dict(torch.load(load_encoder, map_location=device))
        net.encoder = sup.encoder
        train_log.info(f'Model encoder loaded from {load_encoder}')

    net.to(device=device)

    if torch.cuda.device_count() > 1 and device.type != 'cpu':
        net = nn.DataParallel(net)
        train_log.info(f'torch.cuda.device_count:{torch.cuda.device_count()}, Use nn.DataParallel')

    return net


def train_net(net,
              device,
              epochs=50,
              batch_size=128,
              lr=0.0001,
              img_size=51,
              save_cp=True,
              load_optim=False,
              load_scheduler=False,
              dir_checkpoint='details/checkpoints/AUROC',
              dir_img='/nfs3-p1/zsxm/dataset/aorta_classify_ct/',
              flag_3d=False,
              info='',
              **kwargs):

    args = SimpleNamespace(**kwargs)
    module = net.module if isinstance(net, nn.DataParallel) else net
    
    os.makedirs(dir_checkpoint, exist_ok=True)
    dir_checkpoint = os.path.join(dir_checkpoint, train_log.train_time_str + '/')
    writer = SummaryWriter(log_dir=f'details/runs/{train_log.train_time_str}_{module.net_name}_LR_{lr}_BS_{batch_size}_ImgSize_{img_size}')

    if flag_3d:
        train_transform = T.Compose([
            MT.Resize3D(img_size),
            MT.CenterCrop3D(img_size), 
            T.RandomChoice([MT.RandomHorizontalFlip3D(), MT.RandomVerticalFlip3D()]),
            T.RandomApply([MT.ColorJitter3D(0.4, 0.4, 0.4, 0.1)], p=0.7),
            T.RandomApply([MT.RandomRotation3D(45, T.InterpolationMode.BILINEAR)], p=0.4),
            MT.ToTensor3D(), 
        ])
        val_transform = T.Compose([
            MT.Resize3D(img_size),
            MT.CenterCrop3D(img_size),
            MT.ToTensor3D(),
        ])
    else:
        train_transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.RandomChoice([T.RandomHorizontalFlip(), T.RandomVerticalFlip()]),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.7),
            T.RandomApply([T.RandomRotation(45, T.InterpolationMode.BILINEAR)], p=0.4),
            T.ToTensor(),
            #MT.GaussianResidual(3, 1, False),
        ])
        val_transform = T.Compose([
            T.Resize(img_size), # 缩放图片(Image)，保持长宽比不变，最短边为img_size像素
            T.CenterCrop(img_size), # 从图片中间切出img_size*img_size的图片
            T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
            #MT.GaussianResidual(3, 1, False),
            #T.Normalize(mean=[.5], std=[.5]) # 标准化至[-1, 1]，规定均值和标准差
        ])

    if flag_3d:
        train = AortaDataset3DCenter(os.path.join(dir_img, 'train'), transform=train_transform, depth=args.depth_3d, step=args.step_3d, residual=args.residual_3d)
        val = AortaDataset3DCenter(os.path.join(dir_img, 'val'), transform=val_transform, depth=args.depth_3d, step=args.step_3d, residual=args.residual_3d)
    else:
        train = ImageFolder(os.path.join(dir_img, 'train'), transform=train_transform, loader=lambda path: Image.open(path))
        val = ImageFolder(os.path.join(dir_img, 'val'), transform=val_transform, loader=lambda path: Image.open(path))
    
    lsampler = None#LabelSampler(train)
    n_train = len(train) #len(lsampler)
    n_val = len(val)
    train_loader = DataLoader(train, batch_size=batch_size, sampler=lsampler, shuffle=lsampler is None, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    train_log.info(f'''Starting training net:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {img_size}
        Image source:    {dir_img}
        Training info:   {info}
    ''')

    if module.n_classes > 1:
        raise ValueError('n_classes should be 1')
    else:
        tg = torch.tensor(train.targets)
        imratio = ((tg == 1).float().sum() / (tg == 0).float().sum()).item()
        train_log.info(f'imratio = {imratio}')
        criterion = AUCMLoss(imratio=imratio)

    if args.optimizer.lower() == 'pesg':
        optimizer = PESG(net, a=criterion.a, b=criterion.b, alpha=criterion.alpha, imratio=imratio, lr=lr)
    else:
        raise NotImplementedError(f'optimizer not supported: {args.optimizer}')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.1, cooldown=1, min_lr=1e-8, verbose=True)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    if load_optim:
        optimizer.load_state_dict(torch.load(load_optim, map_location=device))
    if load_scheduler:
        scheduler.load_state_dict(torch.load(load_scheduler, map_location=device))

    global_step = 0
    best_val_score = -1 #float('inf') if module.n_classes > 1 else -1
    useless_epoch_count = 0
    for epoch in range(epochs):
        try:
            net.train()
            epoch_loss = 0
            true_list = []
            pred_list = []
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for imgs, true_categories in train_loader:
                    global_step += 1
                    assert imgs.shape[1] == module.n_channels, \
                        f'Network has been defined with {module.n_channels} input channels, ' \
                        f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    category_type = torch.float32 if module.n_classes == 1 else torch.long
                    true_categories = true_categories.to(device=device, dtype=category_type)
                    true_list += true_categories.tolist()

                    categories_pred = F.sigmoid(net(imgs))
                    pred_list += (categories_pred.detach().squeeze(1) > 0.5).float().tolist()
                    loss = criterion(categories_pred, true_categories.unsqueeze(1))
                    epoch_loss += loss.item() * imgs.size(0)
                    writer.add_scalar('Loss/train', loss.item(), global_step)

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])

            train_log.info('Train epoch {} loss: {}'.format(epoch + 1, epoch_loss / n_train))
            train_log.info(f'Train epoch {epoch + 1} train report:\n'+metrics.classification_report(true_list, pred_list, digits=4))

            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                if value.grad is not None:
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            val_score, val_loss = eval_net(net, val_loader, n_val, device)
            scheduler.step(val_score)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if module.n_classes > 1:
                train_log.info('Validation cross entropy: {}'.format(val_loss))
                writer.add_scalar('Loss/val', val_loss, global_step)
                train_log.info('Validation mean Average Precision(mAP): {}'.format(val_score))
                writer.add_scalar('mAP/val', val_score, global_step)
            else:
                train_log.info('Validation binary cross entropy: {}'.format(val_loss))
                writer.add_scalar('Loss/val', val_loss, global_step)
                train_log.info('Validation Area Under roc Curve(AUC): {}'.format(val_score))
                writer.add_scalar('AUC/val', val_score, global_step)
            
            if not flag_3d and (module.n_channels == 1 or module.n_channels == 3):
                writer.add_images('images/origin', imgs, global_step)
            if module.n_classes == 1:
                writer.add_images('categories/true', true_categories[:, None, None, None].repeat(1,1,100,100).float(), global_step)
                writer.add_images('categories/pred', (torch.sigmoid(categories_pred)>0.5)[:, :, None, None].repeat(1,1,100,100), global_step)
            else:
                color_list = [torch.ByteTensor([0,0,0]), torch.ByteTensor([255,0,0]), torch.ByteTensor([0,255,0]), torch.ByteTensor([0,0,255])]
                true_categories_img = torch.zeros(true_categories.shape[0], 100, 100, 3, dtype = torch.uint8)
                categories_pred_img = torch.zeros(categories_pred.shape[0], 100, 100, 3, dtype = torch.uint8)
                categories_pred_idx = categories_pred.argmax(dim=1)
                for category in range(1, module.n_classes):
                    true_categories_img[true_categories==category] = color_list[category]
                    categories_pred_img[categories_pred_idx==category] = color_list[category]
                writer.add_images('categories/true', true_categories_img, global_step, dataformats='NHWC')
                writer.add_images('categories/pred', categories_pred_img, global_step, dataformats='NHWC')

            if val_score > best_val_score: #val_score < best_val_score if module.n_classes > 1 else val_score > best_val_score:
                best_val_score = val_score
                if not os.path.exists(dir_checkpoint):
                    os.makedirs(dir_checkpoint)
                    train_log.info('Created checkpoint directory')
                torch.save(module.state_dict(), dir_checkpoint + 'Net_best.pth')
                train_log.info('Best model saved !')
                useless_epoch_count = 0
            else:
                useless_epoch_count += 1
            
            if save_cp:
                if not os.path.exists(dir_checkpoint):
                    os.makedirs(dir_checkpoint)
                    train_log.info('Created checkpoint directory')
                torch.save(module.state_dict(), dir_checkpoint + f'Net_epoch{epoch + 1}.pth')
                train_log.info(f'Checkpoint {epoch + 1} saved !')

            if args.early_stopping and useless_epoch_count == args.early_stopping:
                train_log.info(f'There are {useless_epoch_count} useless epochs! Early Stop Training!')
                break
        except KeyboardInterrupt:
            train_log.info('Receive KeyboardInterrupt, stop training...')
            break

    torch.save(optimizer.state_dict(), dir_checkpoint + f'Optimizer.pth')
    torch.save(scheduler.state_dict(), dir_checkpoint + f'lrScheduler.pth')
    if not save_cp:
        torch.save(module.state_dict(), dir_checkpoint + 'Net_last.pth')
        train_log.info('Last model saved !')

    # print PR-curve
    train_log.info('Train done! Eval best net and draw PR-curve...')
    args.load_model = os.path.join(dir_checkpoint, 'Net_best.pth')
    args.flag_3d = flag_3d
    net = create_net(device, **vars(args))
    val_score, val_loss, PR_curve_img = eval_net(net, val_loader, n_val, device, final=True, PR_curve_save_dir=dir_checkpoint)
    writer.add_images('PR-curve', np.array(Image.open(PR_curve_img)), dataformats='HWC')
    if module.n_classes > 1:
        train_log.info('Validation cross entropy: {}'.format(val_loss))
        train_log.info('Validation mean Average Precision(mAP): {}'.format(val_score))
    else:
        train_log.info('Validation binary cross entropy: {}'.format(val_loss))
        train_log.info('Validation Area Under roc Curve(AUC): {}'.format(val_score))
    train_log.info(info)

    writer.close()
    return dir_checkpoint




if __name__ == '__main__':
    with open('./args.yaml') as f:
        args = yaml.safe_load(f)
        f.seek(0)
        train_log.info('args.yaml START************************\n'
            f'{f.read()}\n'
            '************************args.yaml END**************************\n')
    
    if str(args['device']) == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args['device'])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.pop('device')
    train_log.info(f'Using device {device}')

    if args['method'].lower() == 'auroc':
        net = create_net(device, **args)
    else:
        raise NotImplementedError(f'method not supported: {args["method"]}')

    try:
        if args['method'].lower() == 'auroc':
            train_net(net, device, **args)
        else:
            raise NotImplementedError(f'method not supported: {args["method"]}')
    except KeyboardInterrupt:
        module = net.module if isinstance(net, nn.DataParallel) else net
        torch.save(module.state_dict(), f'details/checkpoints/{module.__class__.__name__}_{time.strftime("%m-%d_%H:%M:%S", time.localtime())}_INTERRUPTED.pth')
        train_log.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)