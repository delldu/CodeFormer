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

def get_affine_matrix(landmarks, std_landmarks):
    '''min ||Q@M - S||, Q@M ===== S'''
    Q = torch.zeros((10, 4)).to(landmarks.device)

    S = std_landmarks.to(torch.float32).view(-1)
    for i in range(5):
        x, y = landmarks[i]
        Q[i * 2 + 0] = torch.Tensor([x,  y, 1.0, 0.0]).to(landmarks.device)
        Q[i * 2 + 1] = torch.Tensor([y, -x, 0.0, 1.0]).to(landmarks.device)

    M = torch.linalg.lstsq(Q, S).solution.view(-1)
    matrix = torch.Tensor([
        [M[0], M[1], M[2]],
        [-M[1], M[0], M[3]],
        [0.0, 0.0, 1.0]
    ]).to(landmarks.device)

    # ==> matrix @ landmarks[i].view(3, 1）-- stdlandmaks

    return matrix

def get_affine_image(image, matrix, OH, OW):
    '''Sample from image to new image -- output size is (OHxOW)'''

    B, C, H, W = image.shape
    T1 = torch.Tensor([
        [2.0/W,   0.0,   -1.0],
        [0.0,   2.0/H,   -1.0],
        [0.0,     0.0,    1.0]
    ]).to(matrix.device)
    T2 = torch.Tensor([
        [2.0/OW,   0.0,   -1.0],
        [0.0,   2.0/OH,   -1.0],
        [0.0,     0.0,    1.0]
    ]).to(matrix.device)

    theta = torch.linalg.inv(T2 @ matrix @ torch.linalg.inv(T1))
    theta = theta[0:2, :].view(-1, 2, 3)

    grid = F.affine_grid(theta, size=[B, C, OH, OW])
    output = F.grid_sample(image.to(torch.float32), grid, mode='bilinear', padding_mode='zeros')

    return output


def image_mask_erode(bin_img, ksize=7):
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

        kernel = torch.Tensor([[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]).unsqueeze(0).unsqueeze(0)
        pad = 5
        face_mask = torch.ones((1, 1, STANDARD_FACE_SIZE - 4*pad, STANDARD_FACE_SIZE - 4*pad))
        face_mask = F.pad(face_mask, [pad, pad, pad, pad], mode='constant', value=0.5)
        face_mask = F.pad(face_mask, [pad, pad, pad, pad], mode='constant', value=0.0)
        face_mask = F.conv2d(face_mask, kernel, padding=2, groups=1)
        self.face_mask = nn.Parameter(data=face_mask, requires_grad=False)

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

        bg = self.bgzoom2x(x)
        B, C, H, W = bg.size()
        face_locations = self.facedet(bg)  # bbox(4)-score(1)-landm(10), size() -- [2, 15]

        for i in range(face_locations.size(0)):
            landmark = face_locations[i, 5:].view(-1, 2) # skip bbox, score

            eye_dist = torch.abs(landmark[0,0] - landmark[1,0]).item()
            if eye_dist < 5.0: # Skip strange face ...
                continue

            M = get_affine_matrix(landmark, torch.Tensor(STANDARD_FACE_LANDMARKS).to(x.device))
            cropped_face = get_affine_image(bg, M, STANDARD_FACE_SIZE, STANDARD_FACE_SIZE)
            refined_face = self.facegan(cropped_face)[0]

            RM = torch.linalg.inv(M) # get_affine_matrix(torch.Tensor(STANDARD_FACE_LANDMARKS).to(x.device), landmark)
            pasted_face = get_affine_image(refined_face, RM, H, W)
            pasted_mask = get_affine_image(self.face_mask, RM, H, W)

            bg = (1.0 - pasted_mask) * bg + pasted_mask * pasted_face

        return bg
