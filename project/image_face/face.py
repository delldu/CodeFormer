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
import os
import torch
from torch import nn as nn
from torch.nn import functional as F

# from torchvision.transforms.functional import normalize
import todos

from . import rrdbnet, facedet, facegan
import pdb


def load_facegan(model, path, subkey=None):
    """Load model."""
    if not os.path.exists(path):
        raise IOError(f"Model checkpoint '{path}' doesn't exist.")

    # state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    if subkey is not None:
        state_dict = state_dict[subkey]

    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n.startswith("fuse_convs_dict."):
            continue

        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def load_facedet(model, path, subkey=None):
    """Load model."""
    if not os.path.exists(path):
        raise IOError(f"Model checkpoint '{path}' doesn't exist.")

    # state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    if subkey is not None:
        state_dict = state_dict[subkey]

    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        n1 = n.replace("module.", "")
        if n1 in target_state_dict.keys():
            target_state_dict[n1].copy_(p)
        else:
            raise KeyError(n)


def get_affine_matrix(landmarks, std_landmarks):
    """min ||Q@M - S||, Q@M ===== S"""
    Q = torch.zeros((10, 4)).to(landmarks.device)

    S = std_landmarks.to(torch.float32).view(-1)
    for i in range(5):
        # x, y = landmarks[i]
        x = float(landmarks[i, 0])
        y = float(landmarks[i, 1])
        Q[i * 2 + 0] = torch.tensor([x, y, 1.0, 0.0]).to(landmarks.device)
        Q[i * 2 + 1] = torch.tensor([y, -x, 0.0, 1.0]).to(landmarks.device)

    M = torch.linalg.lstsq(Q, S).solution.view(-1)
    matrix = torch.tensor(
        [[float(M[0]), float(M[1]), float(M[2])], [float(-M[1]), float(M[0]), float(M[3])], [0.0, 0.0, 1.0]]
    ).to(landmarks.device)

    # ==> matrix @ landmarks[i].view(3, 1）-- stdlandmaks

    return matrix


def get_affine_image(image, matrix, OH: int, OW: int):
    """Sample from image to new image -- output size is (OHxOW)"""

    B, C, H, W = image.shape
    T1 = torch.tensor([[2.0 / W, 0.0, -1.0], [0.0, 2.0 / H, -1.0], [0.0, 0.0, 1.0]]).to(matrix.device)
    T2 = torch.tensor([[2.0 / OW, 0.0, -1.0], [0.0, 2.0 / OH, -1.0], [0.0, 0.0, 1.0]]).to(matrix.device)

    theta = torch.linalg.inv(T2 @ matrix @ torch.linalg.inv(T1))
    theta = theta[0:2, :].view(-1, 2, 3)

    grid = F.affine_grid(theta, size=[B, C, OH, OW], align_corners=False)
    return F.grid_sample(image.to(torch.float32), grid, mode="bilinear", padding_mode="zeros", align_corners=False)


# def image_mask_erode(bin_img, ksize=7):
#     if ksize % 2 == 0:
#         ksize = ksize + 1

#     B, C, H, W = bin_img.shape
#     pad = (ksize - 1) // 2
#     bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

#     patches = bin_img.unfold(dimension=2, size=ksize, step=1)
#     patches = patches.unfold(dimension=3, size=ksize, step=1)
#     # B x C x H x W x k x k

#     patches = patches.reshape(B, C, H, W, -1)
#     # B x C x H x W x k x k

#     eroded, indices = patches.min(dim=-1)

#     return eroded


class FaceModel(nn.Module):
    """Common Face Model."""

    def __init__(self):
        super(FaceModel, self).__init__()
        # standard 5 landmarks for FFHQ faces with 512 x 512
        self.STANDARD_FACE_SIZE = 512
        self.STANDARD_FACE_LANDMARKS = [
            [192.98138, 239.94708],
            [318.90277, 240.1936],
            [256.63416, 314.01935],
            [201.26117, 371.41043],
            [313.08905, 371.15118],
        ]

        kernel = (
            torch.Tensor(
                [
                    [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                    [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                    [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                    [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                    [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        pad = 5
        face_mask = torch.ones((1, 1, self.STANDARD_FACE_SIZE - 4 * pad, self.STANDARD_FACE_SIZE - 4 * pad))
        face_mask = F.pad(face_mask, [pad, pad, pad, pad], mode="constant", value=0.5)
        face_mask = F.pad(face_mask, [pad, pad, pad, pad], mode="constant", value=0.0)
        face_mask = F.conv2d(face_mask, kernel, padding=2, groups=1)
        self.face_mask = nn.Parameter(data=face_mask, requires_grad=False)

        self.facedet = facedet.RetinaFace()
        self.facegan = facegan.CodeFormer()
        self.bgzoom2x = rrdbnet.RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.min_eyes_distance = 5.0
        # self.load_weights()

        # torch.save(self.state_dict(), "/tmp/image_face_beautify.pth")
        # torch.jit.script(self.facedet) ==> OK
        # torch.jit.script(self.facegan) ==> OK
        # torch.jit.script(self.bgzoom2x) ==> OK

    def load_weights(self):
        # loadnet = torch.load("../weights/CodeFormer/codeformer.pth", map_location=torch.device("cpu"))
        # self.facegan.load_state_dict(loadnet["params_ema"], strict=True)
        load_facegan(self.facegan, "../weights/CodeFormer/codeformer.pth", subkey="params_ema")

        # from copy import deepcopy
        # loadnet = torch.load("../weights/facelib/detection_Resnet50_Final.pth", map_location=torch.device("cpu"))
        # for k, v in deepcopy(loadnet).items():
        #     if k.startswith("module."):
        #         loadnet[k[7:]] = v
        #         loadnet.pop(k)
        # self.facedet.load_state_dict(loadnet, strict=False) # ignal body.layer4
        load_facedet(self.facedet, "../weights/facelib/detection_Resnet50_Final.pth")

        loadnet = torch.load("../weights/realesrgan/RealESRGAN_x2plus.pth", map_location=torch.device("cpu"))
        self.bgzoom2x.load_state_dict(loadnet["params_ema"], strict=True)

    def forward(self, x):
        pass
        return x


class BeautyModel(FaceModel):
    """Beauty Model."""

    def __init__(self):
        super(BeautyModel, self).__init__()

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
            landmark = face_locations[i, 5:].view(-1, 2)  # skip bbox, score

            eye_dist = torch.abs(landmark[0, 0] - landmark[1, 0]).item()
            if eye_dist < self.min_eyes_distance:  # Skip strange face ...
                continue

            M = get_affine_matrix(landmark, torch.tensor(self.STANDARD_FACE_LANDMARKS).to(x.device))
            cropped_face = get_affine_image(bg, M, self.STANDARD_FACE_SIZE, self.STANDARD_FACE_SIZE)
            refined_face = self.facegan(cropped_face)

            RM = torch.linalg.inv(
                M
            )  # get_affine_matrix(torch.Tensor(self.STANDARD_FACE_LANDMARKS).to(x.device), landmark)
            pasted_face = get_affine_image(refined_face, RM, H, W)
            pasted_mask = get_affine_image(self.face_mask, RM, H, W)

            bg = (1.0 - pasted_mask) * bg + pasted_mask * pasted_face

        return bg


class DetectModel(FaceModel):
    """Detect Model."""

    def __init__(self):
        super(DetectModel, self).__init__()

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

        faces = []
        for i in range(face_locations.size(0)):
            landmark = face_locations[i, 5:].view(-1, 2)  # skip bbox, score

            eye_dist = torch.abs(landmark[0, 0] - landmark[1, 0]).item()
            if eye_dist < self.min_eyes_distance:  # Skip strange face ...
                continue

            M = get_affine_matrix(landmark, torch.tensor(self.STANDARD_FACE_LANDMARKS).to(x.device))
            cropped_face = get_affine_image(bg, M, self.STANDARD_FACE_SIZE, self.STANDARD_FACE_SIZE)
            refined_face = self.facegan(cropped_face)

            faces.append(cropped_face)
            faces.append(refined_face)

        if len(faces) < 1:  # NOT Found Face !!!
            return F.interpolate(x, size=[self.STANDARD_FACE_SIZE, self.STANDARD_FACE_SIZE])

        return torch.cat(faces, dim=0)
