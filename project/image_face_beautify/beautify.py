"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 13日 星期二 00:22:40 CST
# ***
# ************************************************************************************/
#

import torch
from torch import nn as nn
from torch.nn import functional as F

from . import rrdbnet, facedet, facegan
import pdb


class BeautyModel(nn.Module):
    """Beauty Model."""

    def __init__(self):
        super(BeautyModel, self).__init__()

        self.facedet = facedet.RetinaFace()
        self.facegan = facegan.CodeFormer()
        self.bgzoom2x = rrdbnet.RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

        self.load_weights()
        # torch.save(self.state_dict(), "/tmp/image_face_beautify.pth")

    def load_weights(self):
        loadnet = torch.load("../weights/CodeFormer/codeformer.pth", map_location=torch.device("cpu"))
        self.facegan.load_state_dict(loadnet["params_ema"], strict=True)

        from copy import deepcopy

        loadnet = torch.load("../weights/facelib/detection_Resnet50_Final.pth", map_location=torch.device("cpu"))
        for k, v in deepcopy(loadnet).items():
            if k.startswith("module."):
                loadnet[k[7:]] = v
                loadnet.pop(k)
        self.facedet.load_state_dict(loadnet, strict=True)

        loadnet = torch.load("../weights/realesrgan/RealESRGAN_x2plus.pth", map_location=torch.device("cpu"))
        self.bgzoom2x.load_state_dict(loadnet["params_ema"], strict=True)

    def forward(self, x):
        B, C, H, W = x.size()

        # Pad x
        pad_h = 1 if (H % 2 != 0) else 0
        pad_w = 1 if (W % 2 != 0) else 0
        if pad_h + pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")

        x = self.bgzoom2x(x)

        return x
