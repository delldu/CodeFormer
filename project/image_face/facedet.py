"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022-2024 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 13日 星期二 00:22:40 CST
# ***
# ************************************************************************************/
#
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from . import resnet

import math
import todos
import pdb
# from ggml_engine import create_network

def load_facedet(model, path):
    """Load model."""
    cdir = os.path.dirname(__file__)
    path = path if cdir == "" else cdir + "/" + path

    if not os.path.exists(path):
        raise IOError(f"Model checkpoint '{path}' doesn't exist.")

    state_dict = torch.load(path, map_location=torch.device("cpu"))
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        n1 = n.replace("module.", "")
        if n1 in target_state_dict.keys():
            target_state_dict[n1].copy_(p)
        else:
            raise KeyError(n)

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode_boxes(loc, anchor_boxes, lin_scale: float=0.1, exp_scale: float=0.2):
    """Decode locations from predictions using anchor_boxes"""
    boxes = torch.cat(
        (
            anchor_boxes[:, :2] + loc[:, :2] * lin_scale * anchor_boxes[:, 2:],
            anchor_boxes[:, 2:] * torch.exp(loc[:, 2:] * exp_scale),
        ),
        dim=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


def decode_landmarks(landmarks, anchor_boxes, lin_scale: float=0.1):
    """Decode landm from predictions using anchor_boxes"""

    tmp = (
        anchor_boxes[:, 0:2] + landmarks[:, 0:2] * lin_scale * anchor_boxes[:, 2:],
        anchor_boxes[:, 0:2] + landmarks[:, 2:4] * lin_scale * anchor_boxes[:, 2:],
        anchor_boxes[:, 0:2] + landmarks[:, 4:6] * lin_scale * anchor_boxes[:, 2:],
        anchor_boxes[:, 0:2] + landmarks[:, 6:8] * lin_scale * anchor_boxes[:, 2:],
        anchor_boxes[:, 0:2] + landmarks[:, 8:10] * lin_scale * anchor_boxes[:, 2:],
    )
    landms = torch.cat(tmp, dim=1)

    return landms

def make_anchor_boxes(H: int, W:int, S: int, min_size1: int, min_size2: int):
    s_kx1 = 1.0*min_size1/W
    s_ky1 = 1.0*min_size1/H
    s_kx2 = 1.0*min_size2/W
    s_ky2 = 1.0*min_size2/H
    H2 = math.ceil(H/S)
    W2 = math.ceil(W/S)

    anchors: List[float] = []
    for i in range(H2):
        cy = (i + 0.5)*S/H
        for j in range(W2):
            cx = (j + 0.5)*S/W
            anchors += [cx, cy, s_kx1, s_ky1]
            anchors += [cx, cy, s_kx2, s_ky2]

    return torch.tensor(anchors).view(-1, 4)


def get_anchor_boxes(H: int, W: int):
    # H = 351, W = 500
    b1 = make_anchor_boxes(H, W, 8, 16, 32)
    b2 = make_anchor_boxes(H, W, 16, 64, 128)
    b3 = make_anchor_boxes(H, W, 32, 256, 512)
    anchors = torch.cat((b1, b2, b3), dim=0)

    # (Pdb) b1.size() -- [5544, 4]
    # (Pdb) b2.size() -- [1408, 4]
    # (Pdb) b3.size() -- [352, 4]

    return anchors # size() -- [7304, 4]

# def nms(boxes, scores, thresh: float):
#     """NMS"""
#     keep = torchvision.ops.nms(
#         boxes=boxes,
#         scores=scores,  # 1D
#         iou_threshold=thresh,
#     )
#     return keep


def nms(bboxes, scores, threshold: float = 0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep: List[int] = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = int(order.item())
            keep.append(i)
            break
        else:
            i = int(order[0].item())
            keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i].item())  # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())
        inter = (xx2 - xx1).clamp(min=0.0) * (yy2 - yy1).clamp(min=0.0)  # [N-1,]

        iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]
        next_order_index = (iou <= threshold).nonzero().squeeze()
        if next_order_index.numel() == 0:
            break

        order = order[next_order_index + 1]  # +1 update index

    return torch.tensor(keep).to(torch.int64)


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


def conv_bn1x1(inp, oup, stride, leaky: float = 0.0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert in_channel == 256 and out_channel == 256

        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=0.0)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=0.0)
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
        super().__init__()
        # in_channels_list:  [512, 1024, 2048]
        # out_channels:  256
        # leaky = 0.0
        self.output1 = conv_bn1x1(in_channels_list[0], out_channels, stride=1, leaky=0.0)
        self.output2 = conv_bn1x1(in_channels_list[1], out_channels, stride=1, leaky=0.0)
        self.output3 = conv_bn1x1(in_channels_list[2], out_channels, stride=1, leaky=0.0)
        self.merge1 = conv_bn(out_channels, out_channels, leaky=0.0)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=0.0)

    def forward(self, input: List[torch.Tensor]) -> List[torch.Tensor]:
        # todos.debug.output_var("FPN input", input)
        # FPN input is list: len = 3
        #     tensor [item] size: [1, 512, 44, 63], min: 0.0, max: 2.788131, mean: 0.08262
        #     tensor [item] size: [1, 1024, 22, 32], min: 0.0, max: 2.534333, mean: 0.033747
        #     tensor [item] size: [1, 2048, 11, 16], min: 0.0, max: 6.305652, mean: 0.326206
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        # Update output2
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        # Update output1
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        # todos.debug.output_var("FPN output", out)
        # FPN output is list: len = 3
        #     tensor [item] size: [1, 256, 44, 63], min: -0.0, max: 6.100448, mean: 0.307941
        #     tensor [item] size: [1, 256, 22, 32], min: -0.0, max: 8.648984, mean: 0.271562
        #     tensor [item] size: [1, 256, 11, 16], min: -0.0, max: 8.366714, mean: 0.319153
        return out


class ClassHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=3):
        super().__init__()
        assert in_channels == 256 and num_anchors == 2

        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(in_channels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous() # [1, 4, 44, 63 --> [1, 44, 63, 4]

        # tensor [ClassHead out1] size: [1, 4, 44, 63], min: -7.161493, max: 6.567798, mean: -0.098771
        # tensor [ClassHead out2] size: [1, 44, 63, 4], min: -7.161493, max: 6.567798, mean: -0.098771
        # tensor [ClassHead out1] size: [1, 4, 22, 32], min: -5.042127, max: 5.503387, mean: 0.038149
        # tensor [ClassHead out2] size: [1, 22, 32, 4], min: -5.042127, max: 5.503387, mean: 0.038149
        # tensor [ClassHead out1] size: [1, 4, 11, 16], min: -5.83153, max: 6.047755, mean: 0.023791
        # tensor [ClassHead out2] size: [1, 11, 16, 4], min: -5.83153, max: 6.047755, mean: 0.023791

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=3):
        super().__init__()
        assert in_channels == 256 and num_anchors == 2

        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous() # [1, (8), 44, 63] --> [1, 44, 63, (8)]

        # tensor [BboxHead out1] size: [1, 8, 44, 63], min: -4.104426, max: 3.965507, mean: 0.055684
        # tensor [BboxHead out2] size: [1, 44, 63, 8], min: -4.104426, max: 3.965507, mean: 0.055684
        # tensor [BboxHead out1] size: [1, 8, 22, 32], min: -4.950671, max: 5.339179, mean: -0.168055
        # tensor [BboxHead out2] size: [1, 22, 32, 8], min: -4.950671, max: 5.339179, mean: -0.168055
        # tensor [BboxHead out1] size: [1, 8, 11, 16], min: -3.178033, max: 2.587976, mean: -0.646355
        # tensor [BboxHead out2] size: [1, 11, 16, 8], min: -3.178033, max: 2.587976, mean: -0.646354

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=3):
        super().__init__()
        assert in_channels == 256 and num_anchors == 2
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous() # [1, 20, 44, 63] --> [1, 44, 63, 20]

        # tensor [LandmarkHead out1] size: [1, 20, 44, 63], min: -8.360132, max: 8.982738, mean: 0.138524
        # tensor [LandmarkHead out2] size: [1, 44, 63, 20], min: -8.360132, max: 8.982738, mean: 0.138524
        # tensor [LandmarkHead out1] size: [1, 20, 22, 32], min: -10.258643, max: 11.092538, mean: -0.033137
        # tensor [LandmarkHead out2] size: [1, 22, 32, 20], min: -10.258643, max: 11.092538, mean: -0.033137
        # tensor [LandmarkHead out1] size: [1, 20, 11, 16], min: -3.90856, max: 4.454429, mean: 0.148574
        # tensor [LandmarkHead out2] size: [1, 11, 16, 20], min: -3.90856, max: 4.454429, mean: 0.148574

        return out.view(out.shape[0], -1, 10)

def make_class_head(fpn_num=3, in_channels=256, anchor_num=2):
    assert fpn_num == 3 and in_channels == 256 and anchor_num == 2

    classhead = nn.ModuleList()
    for i in range(fpn_num):
        classhead.append(ClassHead(in_channels, anchor_num))
    return classhead


def make_bbox_head(fpn_num=3, in_channels=256, anchor_num=2):
    assert fpn_num == 3 and in_channels == 256 and anchor_num == 2

    bboxhead = nn.ModuleList()
    for i in range(fpn_num):
        bboxhead.append(BboxHead(in_channels, anchor_num))
    return bboxhead


def make_landmark_head(fpn_num=3, in_channels=256, anchor_num=2):
    assert fpn_num == 3 and in_channels == 256 and anchor_num == 2

    landmarkhead = nn.ModuleList()
    for i in range(fpn_num):
        landmarkhead.append(LandmarkHead(in_channels, anchor_num))
    return landmarkhead


class RetinaFace(nn.Module):
    def __init__(self):
        super().__init__()
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

        self.ClassHead = make_class_head(fpn_num=3, in_channels=out_channels)
        self.BboxHead = make_bbox_head(fpn_num=3, in_channels=out_channels)
        self.LandmarkHead = make_landmark_head(fpn_num=3, in_channels=out_channels)

        load_facedet(self, "models/detection_Resnet50_Final.pth")

        # RGB: mean - [0.485, 0.456, 0.406], std - 0.229, 0.224, 0.225
        self.register_buffer("mean_tensor", torch.tensor([0.4078, 0.4588, 0.4823]).view(1, 3, 1, 1))
        # self.half()
        # pdb.set_trace()
        # create_network(self)

    def forward(self, x):
        bgr_image = x[:, [2, 1, 0], :, :]  # change channel from RGB to BGR

        B, C, H, W = bgr_image.shape
        # mean_tensor = torch.tensor([0.4078, 0.4588, 0.4823]).view(1, 3, 1, 1).to(bgr_image.device)
        # mean_tensor = torch.tensor([104., 117., 123.])
        # 0.485, 0.456, 0.406 -- RGB ==> 104(B), 117(G), 123(R) == BGR

        bgr_image = bgr_image - self.mean_tensor
        bgr_image = bgr_image * 255.0
        # ==> bgr_image.dtype -- torch.float32, [-123.0, 151.0]


        # tensor [bgr_image] size: [1, 3, 351, 500], min: -108.986504, max: 126.011002, mean: -26.315453
        out = self.body(bgr_image)
        # out is list: len = 3
        #     tensor [item] size: [1, 512, 44, 63], min: 0.0, max: 2.788131, mean: 0.08262
        #     tensor [item] size: [1, 1024, 22, 32], min: 0.0, max: 2.534333, mean: 0.033747
        #     tensor [item] size: [1, 2048, 11, 16], min: 0.0, max: 6.305652, mean: 0.326206

        # FPN
        fpn = self.fpn(out)
        # fpn is list: len = 3
        #     tensor [item] size: [1, 256, 44, 63], min: -0.0, max: 6.100448, mean: 0.307941
        #     tensor [item] size: [1, 256, 22, 32], min: -0.0, max: 8.648984, mean: 0.271562
        #     tensor [item] size: [1, 256, 11, 16], min: -0.0, max: 8.366714, mean: 0.319153

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]
        # tensor [feature1] size: [1, 256, 44, 63], min: 0.0, max: 5.633877, mean: 0.342905
        # tensor [feature2] size: [1, 256, 22, 32], min: 0.0, max: 5.345377, mean: 0.321881
        # tensor [feature3] size: [1, 256, 11, 16], min: 0.0, max: 3.646523, mean: 0.264832


        outs = []
        for i, head in enumerate(self.BboxHead):
            outs.append(head(features[i]))
        bbox_regressions = torch.cat(outs, dim=1)
        # todos.debug.output_var("bbox_regressions", bbox_regressions)
        # tensor [bbox_regressions] size: [1, 7304, 4], min: -4.950671, max: 5.339179, mean: -0.02128

        outs = []
        for i, head in enumerate(self.ClassHead):
            outs.append(head(features[i]))
        classifications = torch.cat(outs, dim=1)
        # tensor [classifications] size: [1, 7304, 2], min: -7.161493, max: 6.567798, mean: -0.06647

        outs = []
        for i, head in enumerate(self.LandmarkHead):
            outs.append(head(features[i]))
        ldm_regressions = torch.cat(outs, dim=1)
        # todos.debug.output_var("ldm_regressions", ldm_regressions)
        # tensor [ldm_regressions] size: [1, 7304, 10], min: -10.258643, max: 11.092538, mean: 0.105917

        # bbox_regressions.shape -- [1, 26720, 4]
        # classifications.shape -- [1, 26720, 2]
        # ldm_regressions.shape -- [1, 26720, 10]

        conf = F.softmax(classifications, dim=2).squeeze(0)
        # tensor [conf] size: [7304, 2], min: 3e-06, max: 0.999997, mean: 0.5


        loc = bbox_regressions.squeeze(0)
        landmarks = ldm_regressions.squeeze(0)
        # (Pdb) conf.size() -- [26720, 2]
        # (Pdb) loc.size() -- [26720, 4]
        # (Pdb) landmarks.size() -- [26720, 10]

        conf_loc_landmarks = torch.cat((conf, loc, landmarks), dim=1)
        # todos.debug.output_var("conf_loc_landmarks", conf_loc_landmarks)
        # tensor [conf_loc_landmarks] size: [7304, 16], min: -10.258643, max: 11.092538, mean: 0.123378

        return conf_loc_landmarks.unsqueeze(0).unsqueeze(0) # [1, 1, 26720, 16] extend dim for onnx output


def decode_conf_loc_landmarks(conf_loc_landmarks, bgr_image):
    conf_loc_landmarks = conf_loc_landmarks.squeeze(0).squeeze(0) # [1, 1, 26720, 16] ==> [26720, 16]

    B, C, H, W = bgr_image.size()
    anchor_boxes = get_anchor_boxes(H=H, W=W).to(bgr_image.device)  # [26720, 4]

    conf = conf_loc_landmarks[:, 0:2]
    loc = conf_loc_landmarks[:, 2:6]
    landmarks = conf_loc_landmarks[:, 6:16]    
    scores = conf[:, 1]  # [26720]

    boxes = decode_boxes(loc, anchor_boxes, lin_scale=0.1, exp_scale=0.2)
    boxes = boxes * torch.tensor([W, H, W, H], dtype=torch.float32).to(bgr_image.device)
    # boxes.size() -- [26720, 4]

    landmarks = decode_landmarks(landmarks, anchor_boxes, lin_scale=0.1)
    landmarks = landmarks * torch.tensor([W, H] * 5, dtype=torch.float32).to(bgr_image.device)
    # landmarks.size() -- [26720, 10]

    # Filter low scores
    conf_threshold = 0.80
    inds = torch.where(conf[:, 1] > conf_threshold)[0]  #  -- torch.int64
    boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]
    # boxes.size() -- [57, 4], scores.size() -- [57], landmarks.size() -- [57, 10]

    # Sort scores
    order = scores.argsort(descending=True)
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

    # do NMS
    nms_threshold = 0.40
    keep = nms(boxes, scores, nms_threshold)

    boxes, landmarks, scores = boxes[keep], landmarks[keep], scores[keep]

    return torch.cat((boxes, scores[:, None], landmarks), dim=1)  # [2, 15]
