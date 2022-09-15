import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models._utils import IntermediateLayerGetter as IntermediateLayerGetter

from facelib.detection.align_trans import get_reference_facial_points, warp_and_crop_face
from facelib.detection.retinaface.retinaface_net import FPN, SSH, MobileNetV1, make_bbox_head, make_class_head, make_landmark_head
from facelib.detection.retinaface.retinaface_utils import (PriorBox, batched_decode, batched_decode_landm, decode, decode_landm,
                                                 py_cpu_nms)
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_config(network_name):

    cfg_mnet = {
        'name': 'mobilenet0.25',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 32,
        'ngpu': 1,
        'epoch': 250,
        'decay1': 190,
        'decay2': 220,
        'image_size': 640,
        'return_layers': {
            'stage1': 1,
            'stage2': 2,
            'stage3': 3
        },
        'in_channel': 32,
        'out_channel': 64
    }

    cfg_re50 = {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 24,
        'ngpu': 4,
        'epoch': 100,
        'decay1': 70,
        'decay2': 90,
        'image_size': 840,
        'return_layers': {
            'layer2': 1,
            'layer3': 2,
            'layer4': 3
        },
        'in_channel': 256,
        'out_channel': 256
    }

    if network_name == 'mobile0.25':
        return cfg_mnet
    elif network_name == 'resnet50':
        return cfg_re50
    else:
        raise NotImplementedError(f'network_name={network_name}')


class RetinaFace(nn.Module):

    def __init__(self, network_name='resnet50', phase='test'):
        super(RetinaFace, self).__init__()

        cfg = generate_config(network_name)
        # cfg -- {'name': 'Resnet50', 'min_sizes': [[16, 32], [64, 128], [256, 512]], 
        # 'steps': [8, 16, 32], 'variance': [0.1, 0.2], 'clip': False, 
        # 'loc_weight': 2.0, 'gpu_train': True, 'batch_size': 24, 'ngpu': 4, 
        # 'epoch': 100, 'decay1': 70, 'decay2': 90, 'image_size': 840, 
        # 'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3}, 
        # 'in_channel': 256, 'out_channel': 256}

        self.backbone = cfg['name'] # 'Resnet50'

        self.model_name = f'retinaface_{network_name}'
        self.cfg = cfg
        self.phase = phase
        self.scale, self.scale1 = None, None
        self.mean_tensor = torch.tensor([[[[104.]], [[117.]], [[123.]]]]).to(device)
        # self.reference = get_reference_facial_points(default_square=True)
        # Build network.
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            self.body = IntermediateLayerGetter(backbone, cfg['return_layers'])
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=False)
            # cfg['return_layers'] -- {'layer2': 1, 'layer3': 2, 'layer4': 3}

            self.body = IntermediateLayerGetter(backbone, cfg['return_layers'])

        in_channels_stage2 = cfg['in_channel'] # 256
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        # [512, 1024, 2048]

        out_channels = cfg['out_channel'] # 256
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

        self.to(device)
        self.eval()

    def forward_x(self, inputs):
         # inputs.size() -- [1, 3, 640, 1013], range in [-255, 255]
        out = self.body(inputs)
        # len(out), out.keys() -- (3, odict_keys([1, 2, 3]))

        if self.backbone == 'mobilenet0.25' or self.backbone == 'Resnet50':
            out = list(out.values())
        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        tmp = [self.LandmarkHead[i](feature) for i, feature in enumerate(features)]
        ldm_regressions = (torch.cat(tmp, dim=1))

        # bbox_regressions.shape -- torch.Size([1, 26720, 4])
        # classifications.shape -- torch.Size([1, 26720, 2])
        # ldm_regressions.shape -- torch.Size([1, 26720, 10])

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

        # (Pdb) output[0].size() -- [1, 26720, 4]
        # (Pdb) output[1].size() -- [1, 26720, 2]
        # (Pdb) output[2].size() -- [1, 26720, 10]

        return output

    def __detect_faces(self, inputs):
        # get scale
        # inputs -- range [-255.0, 255.0]
        height, width = inputs.shape[2:]
        self.scale = torch.tensor([width, height, width, height], dtype=torch.float32).to(device)
        tmp = [width, height] * 5 # width, height, width, height, width, height, width, height]
        self.scale1 = torch.tensor(tmp, dtype=torch.float32).to(device)

        # forawrd
        inputs = inputs.to(device)
        loc, conf, landmarks = self.forward_x(inputs)

        # (Pdb) loc.size() -- [1, 26720, 4]
        # (Pdb) conf.size() -- [1, 26720, 2]
        # (Pdb) landmarks.size() -- [1, 26720, 10]
        
        # get priorbox
        priorbox = PriorBox(self.cfg, image_size=inputs.shape[2:]) # [640, 1013]
        priors = priorbox.forward().to(device)
        # priors.size() -- [26720, 4]

        return loc, conf, landmarks, priors

    # single image detection
    def transform(self, image):
        # image.shape -- (640, 1013, 3)

        image = image.transpose(2, 0, 1) # (3, 640, 1013)
        image = torch.from_numpy(image).unsqueeze(0)

        return image

    def detect_faces(
        self,
        image,
        conf_threshold=0.8,
        nms_threshold=0.4,
    ):
        """
        xxxx8888
                
        Params:
            imgs: BGR image
        """
        # image.shape -- (640, 1013, 3)

        image = self.transform(image)
        image = image.to(device)

        image = image - self.mean_tensor
        # self.mean_tensor.size() -- [1, 3, 1, 1] --- 104, 117, 123

        loc, conf, landmarks, priors = self.__detect_faces(image)

        # self.cfg['variance'] -- [0.1, 0.2]
        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg['variance'])
        boxes = boxes * self.scale
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landmarks = decode_landm(landmarks.squeeze(0), priors, self.cfg['variance'])
        landmarks = landmarks * self.scale1
        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > conf_threshold)[0]
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

        # sort
        order = scores.argsort()[::-1]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # do NMS
        # boxes.shape -- (77, 4)
        # scores[:, np.newaxis].shape -- (77, 1)
        bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(bounding_boxes, nms_threshold)
        bounding_boxes, landmarks = bounding_boxes[keep, :], landmarks[keep]
        # self.t['forward_pass'].toc()
        # print(self.t['forward_pass'].average_time)
        # import sys
        # sys.stdout.flush()

        # bounding_boxes.shape -- (4, 5)
        # landmarks.shape -- (4, 10)
        return np.concatenate((bounding_boxes, landmarks), axis=1)

