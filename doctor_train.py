import os
import sys
import warnings
import time
import yaml
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn import metrics
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from PIL import Image

from utils.eval import eval_net
from utils.print_log import train_log
from models.resnet3d import resnet3d
from utils.datasets import AortaDataset3DCenter, MaskDataset
from models.SupCon import *
import utils.transforms as MT
from models.losses import GradConstraint, GradIntegral

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
              dir_checkpoint='details/checkpoints/ModelDoctor',
              dir_img='/nfs3-p1/zsxm/dataset/aorta_classify_ct/',
              flag_3d=False,
              info='',
              **kwargs):

    args = SimpleNamespace(**kwargs)
    
    os.makedirs(dir_checkpoint, exist_ok=True)
    dir_checkpoint = os.path.join(dir_checkpoint, train_log.train_time_str + '/')
    writer = SummaryWriter(log_dir=f'details/runs/{train_log.train_time_str}_{net.net_name}_{info}')

    if flag_3d:
        tt_list = [
            MT.Resize3D(img_size),
            MT.CenterCrop3D(img_size), 
            T.RandomChoice([MT.RandomHorizontalFlip3D(), MT.RandomVerticalFlip3D()]),
            T.RandomApply([MT.ColorJitter3D(0.4, 0.4, 0.4, 0.1, apply_idx=list(range(args.depth_3d)))], p=0.7),
            T.RandomApply([MT.RandomRotation3D(45, T.InterpolationMode.BILINEAR)], p=0.4),
            MT.ToTensor3D(),
        ]
        if args.sobel:
            tt_list.append(MT.SobelChannel(3, flag_3d, apply_idx=list(range(args.depth_3d))))
        train_transform = T.Compose(tt_list)
        vt_list = [
            MT.Resize3D(img_size),
            MT.CenterCrop3D(img_size),
            MT.ToTensor3D(),
        ]
        if args.sobel:
            vt_list.append(MT.SobelChannel(3, flag_3d))
        val_transform = T.Compose(vt_list)
    else:
        tt_list = [
            MT.Resize3D(img_size),
            MT.CenterCrop3D(img_size),
            T.RandomChoice([MT.RandomHorizontalFlip3D(), MT.RandomVerticalFlip3D()]),
            T.RandomApply([MT.ColorJitter3D(0.4, 0.4, 0.4, 0.1, apply_idx=[0])], p=0.7),
            T.RandomApply([MT.RandomRotation3D(45, T.InterpolationMode.BILINEAR)], p=0.4),
            MT.ToTensor3D(),
        ]
        if args.sobel:
            tt_list.append(MT.SobelChannel(3, flag_3d=True, apply_idx=[0]))
        train_transform = T.Compose(tt_list)
        vt_list = [
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
        ]
        if args.sobel:
            vt_list.append(MT.SobelChannel(3))
        val_transform = T.Compose(vt_list)

    if flag_3d:
        train = AortaDataset3DCenter(os.path.join(dir_img, 'train'), transform=train_transform, depth=args.depth_3d, step=args.step_3d, residual=args.residual_3d, mask_dir=os.path.join(dir_img,'..','mask'))
        val = AortaDataset3DCenter(os.path.join(dir_img, 'val'), transform=val_transform, depth=args.depth_3d, step=args.step_3d, residual=args.residual_3d)
    else:
        train = MaskDataset(os.path.join(dir_img, 'train'), mask_dir=os.path.join(dir_img, 'mask'), transform=train_transform)
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

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss() # FocalLoss(alpha=1/2) # pos_weight=torch.tensor([0.8]).to(device)

    if args.model_name.lower() == 'resnet':
        if args.noise:
            gi = GradIntegral(net, [net.encoder.layer4[2].conv2, net.encoder.layer4[2].conv1, net.encoder.layer3[5].conv2, net.encoder.layer2[3].conv2])
        if flag_3d:
            gc = GradConstraint(net, [net.encoder.layer4[2].conv2, net.encoder.layer4[2].conv1, net.encoder.layer3[5].conv2, net.encoder.layer2[3].conv2], 
                                [args.channel_path], flag_3d=flag_3d, relu=args.relu)
        else:
            gc = GradConstraint(net, [net.encoder.layer4[2].conv2, net.encoder.layer4[2].conv1, net.encoder.layer3[5].conv2, net.encoder.layer2[3].conv2], 
                                [args.channel_path], relu=args.relu)

    if args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters() if args.entire else net.fc.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(net.parameters() if args.entire else net.fc.parameters(), lr=lr)
    elif args.optimizer.lower() == 'nadam':
        optimizer = optim.NAdam(net.parameters() if args.entire else net.fc.parameters(), lr=lr)
    else:
        raise NotImplementedError(f'optimizer not supported: {args.optimizer}')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.1, cooldown=1, min_lr=1e-8, verbose=True)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    if load_optim:
        optimizer.load_state_dict(torch.load(load_optim, map_location=device))
    if load_scheduler:
        scheduler.load_state_dict(torch.load(load_scheduler, map_location=device))

    global_step = 0
    best_val_score = -1 #float('inf') if net.n_classes > 1 else -1
    useless_epoch_count = 0
    for epoch in range(epochs):
        try:
            net.train()
            if args.noise:
                gi.add_noise()
            gc.add_hook()
            epoch_loss, epoch_cls_loss, epoch_channel_loss, epoch_spatial_loss = 0, 0, 0, 0
            true_list = []
            pred_list = []
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for imgs, true_categories, masks in train_loader:
                    global_step += 1
                    assert imgs.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    masks = masks.to(device=device, dtype=torch.float32)
                    category_type = torch.float32 if net.n_classes == 1 else torch.long
                    true_categories = true_categories.to(device=device, dtype=category_type)
                    true_list += true_categories.tolist()

                    categories_pred = net(imgs)

                    if net.n_classes > 1:
                        pred_list += categories_pred.detach().argmax(dim=1).tolist()
                        loss_cls = criterion(categories_pred, true_categories)
                        loss_channel = gc.loss_channel(categories_pred, true_categories)
                        loss_spatial = gc.loss_spatial(categories_pred, true_categories, masks)
                    else:
                        pred_list += (categories_pred.detach().squeeze(1) > 0).float().tolist()
                        loss_cls = criterion(categories_pred, true_categories.unsqueeze(1))
                        loss_channel, loss_spatial = torch.tensor(0).to(device), torch.tensor(0).to(device)
                    loss = loss_cls + loss_channel * 3e10 + loss_spatial * 8e9

                    epoch_loss += loss.item() * imgs.size(0)
                    epoch_cls_loss += loss_cls.item() * imgs.size(0)
                    epoch_channel_loss += loss_channel.item() * imgs.size(0)
                    epoch_spatial_loss += loss_spatial.item() * imgs.size(0)
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    writer.add_scalar('Loss/cls_train', loss_cls.item(), global_step)
                    writer.add_scalar('Loss/channel_train', loss_channel.item(), global_step)
                    writer.add_scalar('Loss/spatial_train', loss_spatial.item(), global_step)

                    pbar.set_postfix(OrderedDict(**{'loss (batch)': loss.item(), 'cls':loss_cls.item(), 'channel': loss_channel.item(), 'spatial':loss_spatial.item()}))

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])

            if args.noise:
                gi.remove_noise()
            gc.remove_hook()

            train_log.info(f'Train epoch {epoch + 1} loss: {epoch_loss/n_train}, cls: {epoch_cls_loss/n_train}, channel: {epoch_channel_loss/n_train}, spatial:{epoch_spatial_loss/n_train}')
            train_log.info(f'Train epoch {epoch + 1} train report:\n'+metrics.classification_report(true_list, pred_list, digits=4))

            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                if value.grad is not None:
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            val_score, val_loss = eval_net(net, val_loader, n_val, device)
            scheduler.step(val_score)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if net.n_classes > 1:
                train_log.info('Validation cross entropy: {}'.format(val_loss))
                writer.add_scalar('Loss/val', val_loss, global_step)
                train_log.info('Validation mean Average Precision(mAP): {}'.format(val_score))
                writer.add_scalar('mAP/val', val_score, global_step)
            else:
                train_log.info('Validation binary cross entropy: {}'.format(val_loss))
                writer.add_scalar('Loss/val', val_loss, global_step)
                train_log.info('Validation Area Under roc Curve(AUC): {}'.format(val_score))
                writer.add_scalar('AUC/val', val_score, global_step)
            
            if not flag_3d and (net.n_channels == 1 or net.n_channels == 3):
                writer.add_images('images/origin', imgs, global_step)
            if net.n_classes == 1:
                writer.add_images('categories/true', true_categories[:, None, None, None].repeat(1,1,100,100).float(), global_step)
                writer.add_images('categories/pred', (torch.sigmoid(categories_pred)>0.5)[:, :, None, None].repeat(1,1,100,100), global_step)
            else:
                color_list = [torch.ByteTensor([0,0,0]), torch.ByteTensor([255,0,0]), torch.ByteTensor([0,255,0]), torch.ByteTensor([0,0,255])]
                true_categories_img = torch.zeros(true_categories.shape[0], 100, 100, 3, dtype = torch.uint8)
                categories_pred_img = torch.zeros(categories_pred.shape[0], 100, 100, 3, dtype = torch.uint8)
                categories_pred_idx = categories_pred.argmax(dim=1)
                for category in range(1, net.n_classes):
                    true_categories_img[true_categories==category] = color_list[category]
                    categories_pred_img[categories_pred_idx==category] = color_list[category]
                writer.add_images('categories/true', true_categories_img, global_step, dataformats='NHWC')
                writer.add_images('categories/pred', categories_pred_img, global_step, dataformats='NHWC')

            if val_score > best_val_score: #val_score < best_val_score if net.n_classes > 1 else val_score > best_val_score:
                best_val_score = val_score
                if not os.path.exists(dir_checkpoint):
                    os.makedirs(dir_checkpoint)
                    train_log.info('Created checkpoint directory')
                torch.save(net.state_dict(), dir_checkpoint + 'Net_best.pth')
                train_log.info('Best model saved !')
                useless_epoch_count = 0
            else:
                useless_epoch_count += 1
            
            if save_cp:
                if not os.path.exists(dir_checkpoint):
                    os.makedirs(dir_checkpoint)
                    train_log.info('Created checkpoint directory')
                torch.save(net.state_dict(), dir_checkpoint + f'Net_epoch{epoch + 1}.pth')
                train_log.info(f'Checkpoint {epoch + 1} saved !')

            if args.early_stopping and useless_epoch_count == args.early_stopping:
                train_log.info(f'There are {useless_epoch_count} useless epochs! Early Stop Training!')
                break
        except KeyboardInterrupt:
            train_log.info('Receive KeyboardInterrupt, stop training...')
            if args.noise:
                gi.remove_noise()
            gc.remove_hook()
            break

    torch.save(optimizer.state_dict(), dir_checkpoint + f'Optimizer.pth')
    torch.save(scheduler.state_dict(), dir_checkpoint + f'lrScheduler.pth')
    if not save_cp:
        torch.save(net.state_dict(), dir_checkpoint + 'Net_last.pth')
        train_log.info('Last model saved !')

    # print PR-curve
    train_log.info('Train done! Eval best net and draw PR-curve...')
    args.load_model = os.path.join(dir_checkpoint, 'Net_best.pth')
    args.flag_3d = flag_3d
    net = create_net(device, **vars(args))
    val_score, val_loss, PR_curve_img = eval_net(net, val_loader, n_val, device, final=True, PR_curve_save_dir=dir_checkpoint)
    writer.add_images('PR-curve', np.array(Image.open(PR_curve_img)), dataformats='HWC')
    if net.n_classes > 1:
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

    if args['method'].lower() == 'model_doctor':
        net = create_net(device, **args)
    else:
        raise NotImplementedError(f'method not supported: {args["method"]}')

    try:
        if args['method'].lower() == 'model_doctor':
            train_net(net, device, **args)
        else:
            raise NotImplementedError(f'method not supported: {args["method"]}')
    except KeyboardInterrupt:
        torch.save(net.state_dict(), f'details/checkpoints/{net.__class__.__name__}_{time.strftime("%m-%d_%H:%M:%S", time.localtime())}_INTERRUPTED.pth')
        train_log.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)