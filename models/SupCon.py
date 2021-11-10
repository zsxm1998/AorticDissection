import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['resnet', 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'SupConResNet', 'TwoCropTransform']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        n_channels: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetEncoder, self).__init__()
        self.n_channels = n_channels
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(n_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ) if stride == 1 else nn.Sequential(nn.AvgPool2d(stride, stride, ceil_mode=True),
                conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        net_name: str = 'ResNet_custom',
        n_channels: int = 3,
        n_classes: int = 1,
        entire: bool = True,
        encoder: Optional[ResNetEncoder] = None,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        self.n_channels = n_channels if encoder is None else encoder.n_channels
        self.n_classes = n_classes
        self.net_name = net_name
        self.entire = entire
        self.encoder = ResNetEncoder(block, layers, n_channels, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer) if encoder is None else encoder
        self.fc = nn.Linear(512 * block.expansion, n_classes)
        if not entire:
            self.encoder.eval()

    def forward(self, x: Tensor) -> Tensor:
        if self.entire:
            x = self.encoder(x)
        else:
            with torch.no_grad():
                x = self.encoder(x)
            x = x.detach() #这句可不可以去掉？
        return self.fc(x)


def resnet18(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], 'ResNet_18', **kwargs)


def resnet34(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], 'ResNet_34', **kwargs)


def resnet50(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], 'ResNet_50', **kwargs)


def resnet101(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], 'ResNet_101', **kwargs)


def resnet152(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], 'ResNet_152', **kwargs)


def resnet(model_depth: int, **kwargs: Any) -> ResNet:
    assert model_depth in [18, 34, 50, 101, 152]
    if model_depth == 18:
        model = resnet18(**kwargs)
    elif model_depth == 34:
        model = resnet34(**kwargs)
    elif model_depth == 50:
        model = resnet50(**kwargs)
    elif model_depth == 101:
        model = resnet101(**kwargs)
    elif model_depth == 152:
        model = resnet152(**kwargs)
    return model


def resnet_encoder18(**kwargs: Any) -> ResNet:
    return ResNetEncoder(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet_encoder34(**kwargs: Any) -> ResNet:
    return ResNetEncoder(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet_encoder50(**kwargs: Any) -> ResNet:
    return ResNetEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet_encoder101(**kwargs: Any) -> ResNet:
    return ResNetEncoder(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet_encoder152(**kwargs: Any) -> ResNet:
    return ResNetEncoder(Bottleneck, [3, 8, 36, 3], **kwargs)


model_dict = {
    'resnet18': [resnet_encoder18, 512],
    'resnet34': [resnet_encoder34, 512],
    'resnet50': [resnet_encoder50, 2048],
    'resnet101': [resnet_encoder101, 2048],
    'resnet152': [resnet_encoder152, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, n_channels=3, name='resnet34', head='mlp', feat_dim=128, norm_encoder_output=False):
        super(SupConResNet, self).__init__()
        self.n_channels = n_channels
        self.name = name
        self.norm_encoder_output = norm_encoder_output
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(n_channels=n_channels)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        if self.norm_encoder_output:
            feat = F.normalize(self.encoder(x), dim=1)
        else:
            feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]