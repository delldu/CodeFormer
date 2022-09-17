import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from facelib.detection import init_detection_model
from facelib.parsing import init_parsing_model
from facelib.utils.misc import img2tensor, imwrite
import pdb


def get_affine_matrix(landmarks, std_landmarks):
    '''min ||Q@M - S||, Q@M ===== S'''

    Q = torch.zeros((10, 4))

    S = std_landmarks.to(torch.float32).view(-1)
    for i in range(5):
        x, y = landmarks[i]
        Q[i * 2 + 0] = torch.Tensor([x,  y, 1.0, 0.0])
        Q[i * 2 + 1] = torch.Tensor([y, -x, 0.0, 1.0])

    M = torch.linalg.lstsq(Q, S).solution.view(-1)
    matrix = torch.Tensor([
        [M[0], M[1], M[2]],
        [-M[1], M[0], M[3]],
        [0.0, 0.0, 1.0]
    ])

    return matrix


def get_affine_image(image, matrix):
    B, C, H, W = image.shape
    if H < 512 or W < 512:
        # (padding_left,padding_right, padding_top,padding_bottom)
        new_image = F.pad(image, [0, (512 - W) if W < 512 else 0, 0, (512-H) if H < 512 else 0], mode='constant', value=128)
    else:
        new_image = image

    B, C, H, W = new_image.shape
    T = torch.Tensor([
        [2.0/W,   0.0,   -1.0],
        [0.0,   2.0/H,   -1.0],
        [0.0,     0.0,    1.0]
    ]).to(matrix.device)

    # T2 = torch.Tensor([
    #     [2.0/512,   0.0,   -1.0],
    #     [0.0,   2.0/512,   -1.0],
    #     [0.0,     0.0,    1.0]
    # ]).to(matrix.device)

    theta = torch.linalg.inv(T @ matrix @ torch.linalg.inv(T))
    # theta = T2 @ matrix @ torch.linalg.inv(T1)
    # theta = torch.linalg.inv(torch.linalg.inv(T2) @ matrix @ T1)
    theta = theta[0:2, :].view(-1, 2, 3)

    grid = F.affine_grid(theta, size=[B, C, H, W])
    output = F.grid_sample(new_image.to(torch.float32), grid, mode='bilinear', padding_mode='border')

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
    blured = patches.mean(dim=-1)

    return eroded, blured


class FaceRestoreHelper(object):
    """Helper for the face restoration pipeline (base class)."""

    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 det_model='retinaface_resnet50',
                 save_ext='png',
                 use_parse=True,
                 device=None):
        # upscale_factor = 2
        # device = device(type='cuda')
        # self.upscale_factor = upscale_factor
        # the cropped face ratio based on the square face
        self.face_size = (face_size, face_size)

        # standard 5 landmarks for FFHQ faces with 512 x 512 
        # facexlib
        self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                       [201.26117, 371.41043], [313.08905, 371.15118]])

        self.face_template = self.face_template * (face_size / 512.0)
        self.save_ext = save_ext

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # init face detection model
        self.face_det = init_detection_model(det_model, half=False, device=self.device)

        # init face parsing model
        self.use_parse = use_parse
        self.face_parse = init_parsing_model(model_name='parsenet', device=self.device)


    def read_image(self, img):
        """img can be image path or cv2 loaded image."""
        # self.input_img is Numpy array, (h, w, c), BGR, uint8, [0, 255]
        # if isinstance(img, str):
        #     img = cv2.imread(img)

        if np.max(img) > 256:  # 16-bit image
            img = img / 65535 * 255
        if len(img.shape) == 2:  # gray image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # xxxx9999
        elif img.shape[2] == 4:  # BGRA image with alpha channel
            img = img[:, :, 0:3]
        self.input_img = img # BGR | BGRA, uint8

    def get_face_landmarks_5(self, resize=640, eye_dist_threshold=5):
        with torch.no_grad():
            bboxes = self.face_det.detect_faces(self.input_img)

        if bboxes is None or bboxes.shape[0] == 0:
            return 0

        for bbox in bboxes:
            # remove faces with too small eye distance: side faces or too small faces
            eye_dist = np.linalg.norm([bbox[6] - bbox[8], bbox[7] - bbox[9]])
            if eye_dist < eye_dist_threshold:
                continue

            landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            self.all_landmarks_5.append(landmark)
            self.det_faces.append(bbox[0:5])
            
        return len(self.all_landmarks_5)

    def align_warp_face(self, border_mode='constant'):
        """Align and warp faces with face template.
        """
        for idx, landmark in enumerate(self.all_landmarks_5):
            # use 5 landmarks to get affine matrix
            # use cv2.LMEDS method for the equivalence to skimage transform
            # ref: https://blog.csdn.net/yichxi/article/details/115827338
            # affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
            # landmark.shape -- (5,2)
            # self.face_template.shape -- (5, 2)
            # affine_matrix.shape -- (2,3)
            # print("affine_matrix: ", affine_matrix)

            M = get_affine_matrix(torch.from_numpy(landmark), torch.from_numpy(self.face_template))
            # print("M: ", M)

            input_img = self.input_img # (512, 811, 3)

            self.affine_matrices.append(M[0:2, :].numpy())
            # warp and crop faces
            # if border_mode == 'constant':
            #     border_mode = cv2.BORDER_CONSTANT
            # elif border_mode == 'reflect101':
            #     border_mode = cv2.BORDER_REFLECT101
            # elif border_mode == 'reflect':
            #     border_mode = cv2.BORDER_REFLECT

            # self.face_size -- (512, 512)
            # cropped_face = cv2.warpAffine(
            #     input_img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132))  # gray


            # cv2.imwrite("/tmp/face_1.png", cropped_face)

            cropped_face = get_affine_image(torch.from_numpy(input_img.transpose(2, 0, 1)).unsqueeze(0), M)
            cropped_face = cropped_face[:, :, :self.face_size[0], :self.face_size[1]]
            cropped_face = cropped_face.squeeze(0).numpy().transpose(1, 2, 0)

            # cv2.imwrite("/tmp/face_2.png", cropped_face_x)
            # pdb.set_trace()

            self.cropped_faces.append(cropped_face)

            # inverse_affine = cv2.invertAffineTransform(affine_matrix)
            inverse_affine = cv2.invertAffineTransform(M[0:2, :].numpy())
            self.inverse_affine_matrices.append(inverse_affine)


    def add_restored_face(self, face):
        self.restored_faces.append(face)

    def paste_faces_to_input_image(self, upsample_img=None, draw_box=False, face_upsampler=None):
        h, w, _ = self.input_img.shape

        assert len(self.restored_faces) == len(
            self.inverse_affine_matrices), ('length of restored_faces and affine_matrices are different.')
        
        inv_mask_borders = []
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w, h))
            # pdb.set_trace()

            # always use square mask
            mask = np.ones(self.face_size, dtype=np.float32) # [1024, 1024]
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w, h)) # [1622, 1024]
            w_edge = 10 # int(total_face_area**0.5) // 20  # -- 9
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(inv_mask, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
            inv_soft_mask = inv_soft_mask[:, :, None]

            if len(upsample_img.shape) == 2:  # upsample_img is gray image
                upsample_img = upsample_img[:, :, None]

            if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 4:  # alpha channel
                alpha = upsample_img[:, :, 3:]
                upsample_img = inv_soft_mask * inv_restored + (1 - inv_soft_mask) * upsample_img[:, :, 0:3]

                upsample_img = np.concatenate((upsample_img, alpha), axis=2)
            else:
                upsample_img = inv_soft_mask * inv_restored + (1 - inv_soft_mask) * upsample_img

        if np.max(upsample_img) > 256:  # 16-bit image
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)

        return upsample_img

    def clean_all(self):
        self.all_landmarks_5 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
        self.det_faces = []
