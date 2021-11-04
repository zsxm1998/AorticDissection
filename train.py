import os
import sys
import warnings
import time
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

from utils.eval import eval_net
from utils.print_log import train_log
from models.resnet3d import generate_model
from utils.datasets import AortaDataset3D, LabelSampler
from models.SupCon import *

warnings.filterwarnings("ignore")
np.random.seed(63910)
torch.manual_seed(53152)
torch.cuda.manual_seed_all(7987)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # faster convolutions, but more memory



"""************************************************** Cross Entropy **************************************************"""
def create_net(device,
               n_channels=1,
               n_classes=4,
               load_model=False,
               flag_3d=False):
    
    if flag_3d:
        net = generate_model(34, n_channels=n_channels, n_classes=n_classes, conv1_t_size=3)
    else:
        net = resnet50(n_channels=n_channels, n_classes=n_classes)

    train_log.info('**********************************************************************\n'
                 f'Network: {net.net_name}\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n'
                 f'\t3D model: {flag_3d}\n')

    if load_model:
        net.load_state_dict(torch.load(load_model, map_location=device))
        train_log.info(f'Model loaded from {load_model}')

    net.to(device=device)

    if torch.cuda.device_count() > 1:
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
              dir_checkpoint='checkpoints/',
              dir_img='/nfs3-p1/zsxm/dataset/aorta_classify_ct/',
              flag_3d=False,
              info='',
              **kwargs):

    args = SimpleNamespace(**kwargs)
    
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
    dir_checkpoint = os.path.join(dir_checkpoint, time.strftime("%m-%d_%H:%M:%S", time.localtime()) + '/')

    writer = SummaryWriter(comment=f'_LR_{lr}_BS_{batch_size}_ImgSize_{img_size}')

    transform = T.Compose([
        T.Resize(img_size), # 缩放图片(Image)，保持长宽比不变，最短边为img_size像素
        T.CenterCrop(img_size), # 从图片中间切出img_size*img_size的图片
        T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
        #T.Normalize(mean=[.5], std=[.5]) # 标准化至[-1, 1]，规定均值和标准差
    ])

    def pil_loader(path):
        with open(path, 'rb') as f:
            return Image.open(f)

    def get_weight_list(dataset):
        pos_list = []
        neg_list = []
        for i in range(len(dataset)):
            if dataset[i][1] == 0:
                neg_list.append(i)
            elif dataset[i][1] == 1:
                pos_list.append(i)
            else:
                raise ValueError
        
        weight_list = []
        pos_num, neg_num = len(pos_list), len(neg_list)
        for i in range(len(dataset)):
            if i in pos_list:
                weight_list.append(1/pos_num)
            elif i in neg_list:
                weight_list.append(1/neg_num)
            else:
                raise ValueError
        return weight_list

    # dataset = ImageFolder(dir_img, transform=transform, loader=lambda path: Image.open(path))
    # ss = StratifiedShuffleSplit(n_splits=1, test_size=val_percent, random_state=7888)
    # labels = [dataset[i][1] for i in range(len(dataset))]
    # train_idx, val_idx = list(ss.split(np.array(labels)[:,np.newaxis], labels))[0]
    # train = torch.utils.data.Subset(dataset, train_idx)
    # val = torch.utils.data.Subset(dataset, val_idx)
    if flag_3d:
        train = AortaDataset3D(os.path.join(dir_img, 'train'), transform=transform, depth=args.depth_3d, step=args.step_3d, residual=args.residual_3d)
        val = AortaDataset3D(os.path.join(dir_img, 'val'), transform=transform, depth=args.depth_3d, step=args.step_3d, residual=args.residual_3d)
    else:
        train = ImageFolder(os.path.join(dir_img, 'train'), transform=transform, loader=lambda path: Image.open(path))
        val = ImageFolder(os.path.join(dir_img, 'val'), transform=transform, loader=lambda path: Image.open(path))
    
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

    module = net.module if isinstance(net, nn.DataParallel) else net
    # encoder_weight_list = []
    # encoder_bias_list = []
    # for name, param in module.encoder.named_parameters():
    #     if 'bias' in name:
    #         encoder_bias_list.append(param)
    #     else:
    #         encoder_weight_list.append(param)
    # classifier_weight_list = []
    # classifier_bias_list = []
    # for name, param in module.classifier.named_parameters():
    #     if 'bias' in name:
    #         classifier_bias_list.append(param)
    #     else:
    #         classifier_weight_list.append(param)
    # optimizer = optim.RMSprop([{'params':encoder_weight_list,'lr':0.01},
    #                           {'params':encoder_bias_list,'lr':0.01,'weight_decay':0},
    #                           {'params':classifier_weight_list},
    #                           {'params':classifier_bias_list,'weight_decay':0}], lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #optimizer = optim.AdamW(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.1, cooldown=1, min_lr=1e-8, verbose=True)
    if load_optim:
        optimizer.load_state_dict(torch.load(load_optim, map_location=device))
    if load_scheduler:
        scheduler.load_state_dict(torch.load(load_scheduler, map_location=device))

    if module.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss() # FocalLoss(alpha=1/2) # pos_weight=torch.tensor([0.8]).to(device)

    global_step = 0
    best_val_score = -1 #float('inf') if module.n_classes > 1 else -1
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
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

                categories_pred = net(imgs)
                if module.n_classes > 1:
                    loss = criterion(categories_pred, true_categories)
                else:
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

        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
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
        
        if not flag_3d:
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
                os.mkdir(dir_checkpoint)
                train_log.info('Created checkpoint directory')
            torch.save(module.state_dict(), dir_checkpoint + 'Net_best.pth')
            train_log.info('Best model saved !')
        
        if save_cp:
            if not os.path.exists(dir_checkpoint):
                os.mkdir(dir_checkpoint)
                train_log.info('Created checkpoint directory')
            torch.save(module.state_dict(), dir_checkpoint + f'Net_epoch{epoch + 1}.pth')
            train_log.info(f'Checkpoint {epoch + 1} saved !')

    torch.save(optimizer.state_dict(), dir_checkpoint + f'Optimizer.pth')
    torch.save(scheduler.state_dict(), dir_checkpoint + f'lrScheduler.pth')

    # print PR-curve
    train_log.info('Train done! Eval best net and draw PR-curve...')
    net = create_net(device, n_channels=args.n_channels, n_classes=args.n_classes, load_model=os.path.join(dir_checkpoint, 'Net_best.pth'), flag_3d=flag_3d)
    val_score, val_loss, PR_curve_img = eval_net(net, val_loader, n_val, device, final=True, PR_curve_save_dir=dir_checkpoint)
    writer.add_images('PR-curve', np.array(Image.open(PR_curve_img)), dataformats='HWC')
    if module.n_classes > 1:
        train_log.info('Validation cross entropy: {}'.format(val_loss))
        train_log.info('Validation mean Average Precision(mAP): {}'.format(val_score))
    else:
        train_log.info('Validation binary cross entropy: {}'.format(val_loss))
        train_log.info('Validation Area Under roc Curve(AUC): {}'.format(val_score))

    writer.close()
    return dir_checkpoint



"""************************************************** Supervised Contrastive **************************************************"""
def create_supcon(device,
                  n_channels=1,
                  load_model=False,
                  flag_3d=False):

    net = SupConResNet(name='resnet34', head='mlp', feat_dim=128)
    net.n_channels, net.n_classes = n_channels, n_classes
    net.encoder.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    train_log.info('**********************************************************************\n'
                 f'Network: {net.__class__.__name__}\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n'
                 f'\t3D model: {flag_3d}\n')

    if load_model:
        net.load_state_dict(torch.load(load_model, map_location=device))
        train_log.info(f'Model loaded from {load_model}')

    net.to(device=device)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        train_log.info(f'torch.cuda.device_count:{torch.cuda.device_count()}, Use nn.DataParallel')

    return net

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

    if args['method'].lower() == 'ce':
        net = create_net(device, n_channels=args['n_channels'], n_classes=args['n_classes'], load_model=args['load_model'])
    elif args['method'].lower() == 'supcon' or args['method'].lower() == 'simclr':
        pass
    else:
        raise NotImplementedError(args['method'])

    try:
        train_net(net, device, **args)
    except KeyboardInterrupt:
        module = net.module if isinstance(net, nn.DataParallel) else net
        torch.save(net.state_dict(), f'checkpoints/{module.__class__.__name__}_INTERRUPTED.pth')
        train_log.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
