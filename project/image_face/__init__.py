"""Image/Video Face Beautify/Detect Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import math
from tqdm import tqdm
import torch
import todos
from . import face

import pdb

def get_tvm_model():
    """Create beauty model."""

    model_path = "models/image_face.pth"  # Share same model between detection and beauty
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = face.FaceBeautyModel()
    todos.model.load(model, checkpoint)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")

    return model, device


def get_beauty_model():
    """Create beauty model."""

    model_path = "models/image_face.pth"  # Share same model between detection and beauty
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = face.FaceBeautyModel()
    todos.model.load(model, checkpoint)
    model = todos.model.ResizePadModel(model)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_face_beauty.torch"):
        model.save("output/image_face_beauty.torch")

    return model, device


def beauty_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_beauty_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)

        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        predict_tensor = todos.model.forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        B, C, H, W = input_tensor.size()
        output_tensor = todos.data.resize_tensor(input_tensor, H, W)

        todos.data.save_tensor([output_tensor, predict_tensor], output_file)
    todos.model.reset_device()


def get_detect_model():
    """Create beauty model."""

    model_path = "models/image_face.pth"  # Share same model between detection and beauty
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = face.FaceDetectModel()
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_face_detect.torch"):
        model.save("output/image_face_detect.torch")

    return model, device


def detect_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_detect_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)

        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        predict_tensor = todos.model.forward(model, device, input_tensor)  # BBx3x512x512

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        if predict_tensor.size(0) < 2:
            todos.data.save_tensor([predict_tensor], output_file)
        else:
            n_row = math.ceil(math.sqrt(predict_tensor.size(0)))
            if n_row % 2 != 0:
                n_row = n_row + 1
            grid_image = todos.data.grid_image(list(torch.split(predict_tensor, 1, dim=0)), nrow=n_row)
            grid_image.save(output_file)
    todos.model.reset_device()
