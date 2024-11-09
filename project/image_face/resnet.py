import torch
import torch.nn as nn
from typing import List, Optional

import todos
import pdb

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self, in_planes: int, out_planes: int,
        stride=1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert stride == 1 or stride == 2

        self.conv1 = conv1x1(in_planes, out_planes, stride=1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes, stride, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = conv1x1(out_planes, out_planes * 4, stride=1)
        self.bn3 = nn.BatchNorm2d(out_planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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
    ):
        super().__init__()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, out_planes, blocks, stride=1):
        assert block.expansion == 4

        downsample = nn.Sequential(
            conv1x1(self.in_planes, out_planes * block.expansion, stride),
            nn.BatchNorm2d(out_planes * block.expansion),
        )
        layers = []
        layers.append(block(self.in_planes, out_planes, stride, downsample))
        self.in_planes = out_planes * block.expansion
        for _ in range(1, blocks): # blocks === [3, 4,  6, 3]
            layers.append(block(self.in_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x) -> List[torch.Tensor]:
        # todos.debug.output_var("ResNet3Layers input", x)
        # tensor [ResNet3Layers input] size: [1, 3, 351, 500], min: -108.986504, max: 126.011002, mean: -26.315453

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # tensor [x] size: [1, 64, 88, 125], min: 0.0, max: 2.57183, mean: 0.281032

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # output = [x2, x3, x4]
        # todos.debug.output_var("ResNet3Layers output", output)
        # ResNet3Layers output is list: len = 3
        #     tensor [item] size: [1, 512, 44, 63], min: 0.0, max: 2.788131, mean: 0.08262
        #     tensor [item] size: [1, 1024, 22, 32], min: 0.0, max: 2.534333, mean: 0.033747
        #     tensor [item] size: [1, 2048, 11, 16], min: 0.0, max: 6.305652, mean: 0.326206
        
        return [x2, x3, x4]


def resnet50_3layers():
    # for resnet50 with 3 layers -- [x2, x3, x4]
    model = ResNet3Layers(Bottleneck, [3, 4, 6, 3])
    return model
