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
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from . import resnet

# import torchvision.models as models
# from torchvision.models._utils import IntermediateLayerGetter as IntermediateLayerGetter
from typing import List, Dict
from . import resnet

import math
import numpy as np
import pdb


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode_boxes(loc, priors, variances: List[float]):
    """Decode locations from predictions using priors"""

    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landms(pre, priors, variances: List[float]):
    """Decode landm from predictions using priors"""
    tmp = (
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
    )
    landms = torch.cat(tmp, dim=1)
    return landms


def prior_box(H: int, W: int):
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [[math.ceil(H / s), math.ceil(W / s)] for s in steps]

    anchors: List[float] = []
    for k, f in enumerate(feature_maps):
        k_sizes = min_sizes[k]
        for i in range(f[0]):
            for j in range(f[1]):
                for min_size in k_sizes:
                    s_kx = min_size / W
                    s_ky = min_size / H
                    cx = (j + 0.5) * steps[k] / W
                    cy = (i + 0.5) * steps[k] / H
                    anchors += [cx, cy, s_kx, s_ky]

    # return torch.Tensor(anchors).view(-1, 4)
    return torch.tensor(anchors).view(-1, 4)  # torch.jit.script only support torch.tensor


def nms(boxes, scores, thresh: float):
    """NMS"""
    keep = torchvision.ops.nms(
        boxes=boxes,
        scores=scores,  # 1D
        iou_threshold=thresh,
    )
    return keep


# def py_cpu_nms(dets, thresh):
#     """Pure Python NMS baseline."""
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     scores = dets[:, 4]

#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)

#         inds = np.where(ovr <= thresh)[0]
#         order = order[inds + 1]

#     return keep


def conv_bn(inp, oup, stride=1, leaky: float = 0.0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky: float = 0.0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


# def conv_dw(inp, oup, stride, leaky:float=0.1):
#     return nn.Sequential(
#         nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#         nn.BatchNorm2d(inp),
#         nn.LeakyReLU(negative_slope=leaky, inplace=True),
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.LeakyReLU(negative_slope=leaky, inplace=True),
#     )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0.0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0.0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input: List[Tensor]) -> List[Tensor]:
        # names = list(input.keys())
        # input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        # xxxx8888
        return out


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


def make_class_head(fpn_num=3, inchannels=64, anchor_num=2):
    classhead = nn.ModuleList()
    for i in range(fpn_num):
        classhead.append(ClassHead(inchannels, anchor_num))
    return classhead


def make_bbox_head(fpn_num=3, inchannels=64, anchor_num=2):
    bboxhead = nn.ModuleList()
    for i in range(fpn_num):
        bboxhead.append(BboxHead(inchannels, anchor_num))
    return bboxhead


def make_landmark_head(fpn_num=3, inchannels=64, anchor_num=2):
    landmarkhead = nn.ModuleList()
    for i in range(fpn_num):
        landmarkhead.append(LandmarkHead(inchannels, anchor_num))
    return landmarkhead


class RetinaFace(nn.Module):
    def __init__(self, phase="test"):
        super(RetinaFace, self).__init__()
        self.phase = phase
        self.body = resnet.resnet50_3layers()

        in_channels_stage2 = 256
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]  # [512, 1024, 2048]

        out_channels = 256
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = make_class_head(fpn_num=3, inchannels=out_channels)
        self.BboxHead = make_bbox_head(fpn_num=3, inchannels=out_channels)
        self.LandmarkHead = make_landmark_head(fpn_num=3, inchannels=out_channels)

    def forward_x(self, bgr_image) -> List[Tensor]:
        # bgr_image.size() -- [1, 3, 640, 1013]
        # bgr_image.dtype -- torch.float32, [-123.0, 151.0]

        out = self.body(bgr_image)
        # len(out) -- 3

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        # bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        outs = []
        for i, head in enumerate(self.BboxHead):
            outs.append(head(features[i]))
        bbox_regressions = torch.cat(outs, dim=1)

        # classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        outs = []
        for i, head in enumerate(self.ClassHead):
            outs.append(head(features[i]))
        classifications = torch.cat(outs, dim=1)

        # tmp = [self.LandmarkHead[i](feature) for i, feature in enumerate(features)]
        # ldm_regressions = torch.cat(tmp, dim=1)
        outs = []
        for i, head in enumerate(self.LandmarkHead):
            outs.append(head(features[i]))
        ldm_regressions = torch.cat(outs, dim=1)

        # bbox_regressions.shape -- [1, 26720, 4]
        # classifications.shape -- [1, 26720, 2]
        # ldm_regressions.shape -- [1, 26720, 10]

        if self.phase == "train":
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

        # (Pdb) output[0].size() -- [1, 26720, 4]
        # (Pdb) output[1].size() -- [1, 26720, 2]
        # (Pdb) output[2].size() -- [1, 26720, 10]

        return output

    def forward(self, x):
        bgr_image = x[:, [2, 1, 0], :, :]  # change channel from RGB to BGR

        B, C, H, W = bgr_image.shape
        mean_tensor = torch.tensor([0.4078, 0.4588, 0.4823]).view(1, 3, 1, 1).to(bgr_image.device)
        # mean_tensor = torch.tensor([104., 117., 123.])
        # 0.485, 0.456, 0.406 -- RGB ==> 104(B), 117(G), 123(R) == BGR

        bgr_image = bgr_image - mean_tensor
        bgr_image = bgr_image * 255.0
        # ==> bgr_image.dtype -- torch.float32, [-123.0, 151.0]

        loc, conf, landmarks = self.forward_x(bgr_image)
        # (Pdb) loc.size() -- [1, 26720, 4]
        # (Pdb) conf.size() -- [1, 26720, 2]
        # (Pdb) landmarks.size() -- [1, 26720, 10]

        priors = prior_box(H=H, W=W).to(bgr_image.device)  # [26720, 4]

        boxes = decode_boxes(loc.squeeze(0), priors, [0.1, 0.2])
        boxes = boxes * torch.tensor([W, H, W, H], dtype=torch.float32).to(bgr_image.device)
        # boxes.size() -- [28952, 4]

        scores = conf.squeeze(0)[:, 1]  # [28952]
        landmarks = decode_landms(landmarks.squeeze(0), priors, [0.1, 0.2])
        landmarks = landmarks * torch.tensor([W, H] * 5, dtype=torch.float32).to(bgr_image.device)
        # landmarks.size() -- [28952, 10]

        # Filter low scores
        conf_threshold = 0.8
        inds = torch.where(conf.squeeze(0)[:, 1] > conf_threshold)[0]  #  -- torch.int64
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]
        # boxes.size() -- [57, 4], scores.size() -- [57], landmarks.size() -- [57, 10]

        # Sort scores
        order = scores.argsort(descending=True)
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # do NMS
        nms_threshold = 0.4
        keep = nms(boxes, scores, nms_threshold)
        boxes, landmarks, scores = boxes[keep], landmarks[keep], scores[keep]

        return torch.cat((boxes, scores[:, None], landmarks), dim=1)  # [2, 15]
