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
from torchvision.transforms.functional import normalize
import todos

from . import rrdbnet, facedet, facegan
import pdb

STANDARD_FACE_SIZE = 512
# standard 5 landmarks for FFHQ faces with 512 x 512
STANDARD_FACE_LANDMARKS = [
    [192.98138, 239.94708],
    [318.90277, 240.1936],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
]

def get_affine_matrix(src_pts, dst_pts):
    '''min(|AX - B|)'''

    N, C = src_pts.size()
    ones = torch.ones((N, 1)).to(src_pts.device)
    A = torch.hstack([src_pts, ones]).to(torch.float32)
    B = torch.hstack([dst_pts, ones]).to(torch.float32)

    X, res, rank, s = torch.linalg.lstsq(A, B) # np.linalg.lstsq(A, B) ?
    print("X:", X)

    rank = rank.item()

    if rank == 3:
        M = torch.Tensor([
            [X[0, 0], X[1, 0], X[2, 0]],
            [X[0, 1], X[1, 1], X[2, 1]],
            [0.0,      0.0,     1.0],
        ])
    elif rank == 2:
        M = torch.Tensor([
            [X[0, 0], X[1, 0], 0.0],
            [X[0, 1], X[1, 1], 0.0],
            [0.0,      0.0,     1.0],
        ])
    else:
        M = torch.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    return M

def get_affine_image(image, matrix):
    B, C, H, W = image.shape

    T = torch.Tensor([
        [2.0/W,   0.0,   -1.0],
        [0.0,   2.0/H,   -1.0],
        [0.0,     0.0,    1.0]
    ]).to(matrix.device)

    theta = torch.linalg.inv(T @ matrix @ torch.linalg.inv(T))
    theta = theta[0:2, :].view(-1, 2, 3)

    grid = F.affine_grid(theta, size=[B, C, H, W])
    output = F.grid_sample(image.to(torch.float32), grid, mode='bilinear', padding_mode='zeros')

    return output

def image_mask_erode(bin_img, ksize=5):
    if ksize % 2 == 0:
        ksize = ksize + 1

    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    patches = patches.reshape(B, C, H, W, -1)
    # B x C x H x W x k x k

    eroded, indices = patches.min(dim=-1)

    return eroded


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

        # torch.jit.script(self.facedet)

    def forward(self, x):
        B, C, H, W = x.size()

        # Pad x
        pad_h = 1 if (H % 2 != 0) else 0
        pad_w = 1 if (W % 2 != 0) else 0
        if pad_h + pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")

        bg = self.bgzoom2x(x)[:, 0:3, :, :]
        face_locations = self.facedet(bg)  # box(4)-score(1)-landm(10), size() -- [2, 15]

        #          self.all_landmarks_5
        # [array([[204.07124, 227.93121],
        #        [302.27927, 222.11633],
        #        [238.77112, 286.655  ],
        #        [221.4199 , 324.38986],
        #        [310.97925, 317.86227]], dtype=float32), 

        # array([[652.6402 , 267.94696],
        #        [743.5763 , 267.42746],
        #        [694.3073 , 316.09866],
        #        [657.8458 , 350.22675],
        #        [744.43146, 349.51352]], dtype=float32)]

        # (Pdb) face_locations[:, 5:].view(-1, 5, 2)
        # tensor([[[650.7346, 266.6703],
        #          [748.6882, 264.9617],
        #          [695.1119, 313.8882],
        #          [660.6911, 351.7814],
        #          [746.7521, 349.7393]],

        #         [[204.3869, 229.1595],
        #          [301.4592, 223.9881],
        #          [237.4415, 286.2192],
        #          [220.0092, 324.7602],
        #          [310.0208, 318.8951]]], device='cuda:0')



        for i in range(face_locations.size(0)):
            landmark = face_locations[i, 5:].view(-1, 2)
            M = get_affine_matrix(landmark, torch.Tensor(STANDARD_FACE_LANDMARKS).to(x.device))
            face = get_affine_image(bg, M)
            face = face[:, :, :512, :512]

            output_file = f"/tmp/face_{i+1:03d}.png"
            todos.data.save_tensor([face], output_file)

        pdb.set_trace()

        return x
