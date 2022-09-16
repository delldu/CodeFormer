# Modified by Shangchen Zhou from: https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py
import os
import cv2
import argparse
import glob
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
import pdb


def set_realesrgan():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model,
        tile=args.bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=True)  # need to set False in CPU mode

    return bg_upsampler

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    parser.add_argument('--w', type=float, default=0.5, help='Balance the quality and fidelity')
    parser.add_argument('--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--test_path', type=str, default='./inputs/cropped_faces')
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50')
    parser.add_argument('--draw_box', action='store_true')
    parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler. Default: 400')

    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]

    w = args.w
    result_root = f'results/{os.path.basename(args.test_path)}_{w}'

    # ------------------ set up background upsampler ------------------
    bg_upsampler = set_realesrgan()

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)
    
    ckpt_path = 'weights/CodeFormer/codeformer.pth'
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # ckpt_path -- weights/CodeFormer/codeformer.pth
    # net -- CodeFormer(...)

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    print(f'Face detection model: {args.detection_model}')

    face_helper = FaceRestoreHelper(
        1, # xxxx8888 args.upscale, # 2
        face_size=512,
        det_model = args.detection_model, # 'retinaface_resnet50'
        save_ext='png',
        use_parse=True,
        device=device)

    # -------------------- start to processing ---------------------
    # scan all the jpg and png images
    for img_path in sorted(glob.glob(os.path.join(args.test_path, '*.[jp][pn]g'))):
        # clean all the intermediate results to process the next image
        face_helper.clean_all()
        
        img_name = os.path.basename(img_path)
        print(f'Processing: {img_name}')
        basename, ext = os.path.splitext(img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # BGR format

        print(img_path, "img.shape: ", img.shape)

        # upsample the background
        # Now only support RealESRGAN for upsampling background
        bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0] # BGR | BGRA, uint8
        print("bg_img shape: ", bg_img.shape)


        face_helper.read_image(bg_img)
        print("face image ===> ", img_path, "image.shape: ", face_helper.input_img.shape)

        # get face landmarks for each face
        num_det_faces = face_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)

        print(f'\tdetect {num_det_faces} faces')
        # align and warp each face
        face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = net(cropped_face_t)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

                print("face", idx, " cropped_face_t.size():", cropped_face_t.size(), "==>", output.size())

                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face)

        # paste_back


        # face_helper.get_inverse_affine()

        # paste each restored face to the input image
        restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, face_upsampler=bg_upsampler)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
            # save cropped face
            save_crop_path = os.path.join(result_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
            imwrite(cropped_face, save_crop_path)

            # save restored face
            save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(result_root, 'restored_faces', save_face_name)

            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)

            imwrite(cmp_img, save_restore_path)

        # save restored image
        if restored_img is not None:
            save_restore_path = os.path.join(result_root, 'final_results', f'{basename}.png')
            imwrite(restored_img, save_restore_path)

    print(f'\nAll results are saved in {result_root}')
