import math
from functools import partial
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['resnet3d', 'ResNet3D', 'resnet3d_encoder', 'SupConResNet3D']


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetEncoder3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0):
        super().__init__()

        self.n_channels = n_channels

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResNet3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 net_name='ResNet3D_custom',
                 n_channels=3,
                 n_classes=400,
                 entire=True,
                 encoder=None,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0):
        super().__init__()
        self.n_channels = n_channels if encoder is None else encoder.n_channels
        self.n_classes = n_classes
        self.net_name = net_name
        self.entire = entire

        self.encoder = ResNetEncoder3D(block, layers, block_inplanes, n_channels, conv1_t_size, conv1_t_stride, no_max_pool, shortcut_type, widen_factor)
        self.fc = nn.Linear(int(block_inplanes[3]*widen_factor), n_classes)

    def forward(self, x):
        if self.entire:
            x = self.encoder(x)
        else:
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x)
            x = x.detach() #这句可不可以去掉？
        return self.fc(x)


def resnet3d(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet3D(BasicBlock, [1, 1, 1, 1], get_inplanes(), 'ResNet3D_10', **kwargs)
    elif model_depth == 18:
        model = ResNet3D(BasicBlock, [2, 2, 2, 2], get_inplanes(), 'ResNet3D_18', **kwargs)
    elif model_depth == 34:
        model = ResNet3D(BasicBlock, [3, 4, 6, 3], get_inplanes(), 'ResNet3D_34', **kwargs)
    elif model_depth == 50:
        model = ResNet3D(Bottleneck, [3, 4, 6, 3], get_inplanes(), 'ResNet3D_50', **kwargs)
    elif model_depth == 101:
        model = ResNet3D(Bottleneck, [3, 4, 23, 3], get_inplanes(), 'ResNet3D_101', **kwargs)
    elif model_depth == 152:
        model = ResNet3D(Bottleneck, [3, 8, 36, 3], get_inplanes(), 'ResNet3D_152', **kwargs)
    elif model_depth == 200:
        model = ResNet3D(Bottleneck, [3, 24, 36, 3], get_inplanes(), 'ResNet3D_200', **kwargs)

    return model


def resnet3d_encoder18(**kwargs):
    return ResNetEncoder3D(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)

def resnet3d_encoder34(**kwargs):
    return ResNetEncoder3D(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)

def resnet3d_encoder50(**kwargs):
    return ResNetEncoder3D(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)

def resnet3d_encoder101(**kwargs):
    return ResNetEncoder3D(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)

def resnet3d_encoder152(**kwargs):
    return ResNetEncoder3D(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)

def resnet3d_encoder(model_depth, **kwargs):
    assert model_depth in [18, 34, 50, 101, 152], f'model_depth={model_depth}'
    if model_depth == 18:
        model = resnet3d_encoder18(**kwargs)
    elif model_depth == 34:
        model = resnet3d_encoder34(**kwargs)
    elif model_depth == 50:
        model = resnet3d_encoder50(**kwargs)
    elif model_depth == 101:
        model = resnet3d_encoder101(**kwargs)
    elif model_depth == 152:
        model = resnet3d_encoder152(**kwargs)
    return model


model_dict = {
    'resnet18': [resnet3d_encoder18, 512],
    'resnet34': [resnet3d_encoder34, 512],
    'resnet50': [resnet3d_encoder50, 2048],
    'resnet101': [resnet3d_encoder101, 2048],
    'resnet152': [resnet3d_encoder152, 2048],
}


class SupConResNet3D(nn.Module):
    """backbone + projection head"""
    def __init__(self, n_channels=3, name='resnet34', encoder=None, head='mlp', feat_dim=128, norm_encoder_output=False, conv1_t_size=7):
        super(SupConResNet3D, self).__init__()
        self.n_channels = n_channels
        self.name = name
        self.norm_encoder_output = norm_encoder_output
        model_fun, self.dim_in = model_dict[name]
        self.encoder = model_fun(n_channels=n_channels, conv1_t_size=conv1_t_size) if encoder is None else encoder
        if head == 'linear':
            self.head = nn.Linear(self.dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.dim_in, self.dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        r = self.encoder(x)
        if self.norm_encoder_output:
            z = F.normalize(r, dim=1)
            z = F.normalize(self.head(z), dim=1)
        else:
            z = F.normalize(self.head(r), dim=1)
        return z, r