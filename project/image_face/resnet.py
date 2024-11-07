import torch
import torch.nn as nn
from typing import List, Optional
import pdb

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    # assert groups == 1 and dilation == 1
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups,
    #     bias=False,
    #     dilation=dilation,
    # )
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self,
        inplanes,
        planes,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super().__init__()
        # norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

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

    def __init__(self,
        inplanes: int,
        planes: int,
        stride=1,
        downsample: Optional[nn.Module] = None,
        # norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: # True or False
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3Layers(nn.Module):
    def __init__(self,
        block, # Bottleneck
        layers=[3, 4, 6, 3],
        groups=1,
        width_per_group=64,
    ):
        super().__init__()

        # norm_layer = nn.BatchNorm2d
        # self._norm_layer = norm_layer

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        # pdb.set_trace()

    def _make_layer(self, block, planes, blocks, stride=1):
        # norm_layer = self._norm_layer
        downsample = None
            
        if stride != 1 or planes * block.expansion != 64:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x2, x3, x4]


def resnet50_3layers():
    # for resnet50 with 3 layers -- [x2, x3, x4]
    model = ResNet3Layers(Bottleneck, [3, 4, 6, 3])
    return model
