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
from sklearn.model_selection import StratifiedShuffleSplit
#import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from PIL import Image

from utils.eval import eval_combine, eval_net, eval_supcon
from utils.print_log import train_log
from models.resnet3d import SupConResNet3D, resnet3d
from utils.datasets import AortaDataset3D, LabelSampler, AortaDataset3DCenter
from models.SupCon import *
from models.losses import SupConLoss
from utils import transforms as MT

warnings.filterwarnings("ignore")
# np.random.seed(63910)
# torch.manual_seed(53152)
# torch.cuda.manual_seed_all(7987)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # faster convolutions, but more memory



def combine_train(device,
                  model_depth = 34,
                  n_channels = 1,
                  n_classes = 4,
                  epochs=50,
                  batch_size=128,
                  lr=0.0001,
                  img_size=51,
                  save_cp=True,
                  load_optim=False,
                  load_scheduler=False,
                  dir_checkpoint='details/checkpoints/Combine',
                  dir_img='/nfs3-p1/zsxm/dataset/aorta_classify_ct/',
                  flag_3d=False,
                  info='',
                  **kwargs):

    args = SimpleNamespace(**kwargs)

    if flag_3d:
        supcon = SupConResNet3D(n_channels=n_channels, name=f'resnet{model_depth}', head='mlp', feat_dim=128, norm_encoder_output=args.norm_encoder_output, conv1_t_size=3)
    else:
        supcon = SupConResNet(n_channels=n_channels, name=f'resnet{model_depth}', head='mlp', feat_dim=128, norm_encoder_output=args.norm_encoder_output)
    fc = nn.Linear(supcon.dim_in, n_classes)
    supcon.to(device)
    fc.to(device)
    if torch.cuda.device_count() > 1 and device.type != 'cpu':
        supcon = nn.DataParallel(supcon)
        fc = nn.DataParallel(fc)
        train_log.info(f'torch.cuda.device_count:{torch.cuda.device_count()}, Use nn.DataParallel')
    
    supcon_module = supcon.module if isinstance(supcon, nn.DataParallel) else supcon
    fc_module = fc.module if isinstance(fc, nn.DataParallel) else fc
    
    os.makedirs(dir_checkpoint, exist_ok=True)
    dir_checkpoint = os.path.join(dir_checkpoint, train_log.train_time_str + '/')
    writer = SummaryWriter(log_dir=f'details/runs/{train_log.train_time_str}_Combine{model_depth}_LR_{lr}_BS_{batch_size}_ImgSize_{img_size}')

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
        ])
        val_transform = T.Compose([
            T.Resize(img_size), # 缩放图片(Image)，保持长宽比不变，最短边为img_size像素
            T.CenterCrop(img_size), # 从图片中间切出img_size*img_size的图片
            T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
            #T.Normalize(mean=[.5], std=[.5]) # 标准化至[-1, 1]，规定均值和标准差
        ])

    if flag_3d:
        train = AortaDataset3DCenter(os.path.join(dir_img, 'train'), transform=train_transform, depth=args.depth_3d, step=args.step_3d, residual=args.residual_3d, supcon=True)
        val = AortaDataset3DCenter(os.path.join(dir_img, 'val'), transform=val_transform, depth=args.depth_3d, step=args.step_3d, residual=args.residual_3d)
    else:
        train = ImageFolder(os.path.join(dir_img, 'train'), transform=TwoCropTransform(train_transform), loader=lambda path: Image.open(path))
        val = ImageFolder(os.path.join(dir_img, 'val'), transform=val_transform, loader=lambda path: Image.open(path))
    
    lsampler = None#LabelSampler(train)
    n_train = len(train) #len(lsampler)
    n_val = len(val)
    train_loader = DataLoader(train, batch_size=batch_size, sampler=lsampler, shuffle=lsampler is None, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    train_log.info(f'''Starting combine training:
        Epochs:             {epochs}
        Batch size:         {batch_size}
        Learning rate:      {lr}
        Training size:      {n_train}
        Validation size:    {n_val}
        Checkpoints:        {save_cp}
        Device:             {device.type}
        Images size:        {img_size}
        Image source:       {dir_img}
        Training info:      {info}
    ''')

    if args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop([{'params': supcon_module.parameters()}, {'params': fc_module.parameters()}], lr=lr, weight_decay=1e-8, momentum=0.9)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam([{'params': supcon_module.parameters()}, {'params': fc_module.parameters()}], lr=lr)
    elif args.optimizer.lower() == 'nadam':
        optimizer = optim.NAdam([{'params': supcon_module.parameters()}, {'params': fc_module.parameters()}], lr=lr)
    else:
        raise NotImplementedError(f'optimizer not supported: {args.optimizer}')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.1, cooldown=1, min_lr=1e-8, verbose=True)
    if load_optim:
        optimizer.load_state_dict(torch.load(load_optim, map_location=device))
    if load_scheduler:
        scheduler.load_state_dict(torch.load(load_scheduler, map_location=device))

    supcon_criterion = SupConLoss(temperature=args.temperature)
    if n_classes > 1:
        ce_criterion = nn.CrossEntropyLoss()
    else:
        ce_criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    best_val_score = -1
    useless_epoch_count = 0
    for epoch in range(epochs):
        try:
            supcon.train()
            fc.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for imgs, true_categories in train_loader:
                    global_step += 1

                    imgs = torch.cat([imgs[0], imgs[1]], dim=0)
                    assert imgs.shape[1] == n_channels, \
                        f'Network has been defined with {n_channels} input channels, ' \
                        f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                    bsz = true_categories.shape[0]

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    category_type = torch.float32 if n_classes == 1 else torch.long
                    true_categories = true_categories.to(device=device, dtype=category_type)

                    features, representations = supcon(imgs)
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    if args.method.lower() == 'supcon':
                        loss_con = supcon_criterion(features, true_categories)
                    elif args.method.lower() == 'simclr':
                        loss_con = supcon_criterion(features)
                    else:
                        raise ValueError(f'contrastive method not supported: {args.method}')
                    pred_categories = fc(representations)
                    if n_classes == 1:
                        true_categories = true_categories.unsqueeze(1)
                    loss_ce = ce_criterion(pred_categories, torch.cat([true_categories, true_categories], dim=0))
                    loss = args.alpha*loss_ce + (1-args.alpha)*loss_con

                    epoch_loss += loss.item() * bsz
                    writer.add_scalar('Loss/train', loss.item(), global_step)

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(supcon_module.parameters(), 0.1)
                    nn.utils.clip_grad_value_(fc_module.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(bsz)

            epoch_loss /= n_train
            train_log.info('Train epoch {} loss: {}'.format(epoch + 1, epoch_loss))

            for tag, value in supcon_module.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/supcon/' + tag, value.data.cpu().numpy(), global_step)
                writer.add_histogram('grads/suocon/' + tag, value.grad.data.cpu().numpy(), global_step)
            for tag, value in fc_module.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/fc/' + tag, value.data.cpu().numpy(), global_step)
                writer.add_histogram('grads/fc/' + tag, value.grad.data.cpu().numpy(), global_step)
            val_score, val_loss, val_ratio = eval_combine(supcon, fc, val_loader, n_val, device, n_classes)
            scheduler.step(val_score)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if n_classes > 1:
                train_log.info('Validation cross entropy: {}'.format(val_loss))
                writer.add_scalar('Loss/val', val_loss, global_step)
                train_log.info('Validation mean Average Precision(mAP): {}'.format(val_score))
                writer.add_scalar('mAP/val', val_score, global_step)
            else:
                train_log.info('Validation binary cross entropy: {}'.format(val_loss))
                writer.add_scalar('Loss/val', val_loss, global_step)
                train_log.info('Validation Area Under roc Curve(AUC): {}'.format(val_score))
                writer.add_scalar('AUC/val', val_score, global_step)
            train_log.info('Validation inner_dis/outer_dis ratio: {}'.format(val_ratio))
            writer.add_scalar('dis_ratio/val', val_ratio, global_step)
            
            if not flag_3d:
                writer.add_images('images/origin', imgs, global_step)

            if val_score > best_val_score: #val_score < best_val_score if module.n_classes > 1 else val_score > best_val_score:
                best_val_score = val_score
                if not os.path.exists(dir_checkpoint):
                    os.makedirs(dir_checkpoint)
                    train_log.info('Created checkpoint directory')
                torch.save(supcon_module.state_dict(), dir_checkpoint + 'supcon_best.pth')
                torch.save(fc_module.state_dict(), dir_checkpoint + 'fc_best.pth')
                train_log.info('Best model saved !')
                useless_epoch_count = 0
            else:
                useless_epoch_count += 1
            
            if save_cp:
                if not os.path.exists(dir_checkpoint):
                    os.makedirs(dir_checkpoint)
                    train_log.info('Created checkpoint directory')
                torch.save(supcon_module.state_dict(), dir_checkpoint + f'supcon_epoch{epoch + 1}.pth')
                torch.save(fc_module.state_dict(), dir_checkpoint + f'fc_epoch{epoch + 1}.pth')
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
        torch.save(supcon_module.state_dict(), dir_checkpoint + 'supcon_last.pth')
        torch.save(fc_module.state_dict(), dir_checkpoint + 'fc_last.pth')
        train_log.info('Last model saved !')

    # print PR-curve and t-SNE
    train_log.info('Train done! Eval best net and draw PR-curve...')
    supcon_module.load_state_dict(torch.load(os.path.join(dir_checkpoint, 'supcon_best.pth'), map_location=device))
    fc_module.load_state_dict(torch.load(os.path.join(dir_checkpoint, 'fc_best.pth'), map_location=device))
    if torch.cuda.device_count() > 1 and device.type != 'cpu':
        supcon.module, fc.module = supcon_module, fc_module
    else:
        supcon, fc = supcon_module, fc_module
    val_score, val_loss, val_ratio, PR_curve_img, TSNE_img = eval_combine(supcon, fc, val_loader, n_val, device, n_classes, final=True, PR_curve_save_dir=dir_checkpoint, TSNE_save_dir=dir_checkpoint)
    writer.add_images('PR-curve', np.array(Image.open(PR_curve_img)), dataformats='HWC')
    writer.add_images('t-SNE', np.array(Image.open(TSNE_img)), dataformats='HWC')
    if n_classes > 1:
        train_log.info('Validation cross entropy: {}'.format(val_loss))
        train_log.info('Validation mean Average Precision(mAP): {}'.format(val_score))
    else:
        train_log.info('Validation binary cross entropy: {}'.format(val_loss))
        train_log.info('Validation Area Under roc Curve(AUC): {}'.format(val_score))
    train_log.info('Validation inner_dis/outer_dis ratio: {}'.format(val_ratio))
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

    combine_train(device, **args)
