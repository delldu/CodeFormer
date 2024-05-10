"""Image/Video Face Beautify/Detect Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021-2024(18588220928@163.com) All Rights Reserved.
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
from . import facegan
from . import facedet

import pdb


def get_facedet_model():
    model = facedet.RetinaFace()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    if 'cpu' in str(device.type):
        model.float()

    print(f"Running on {device} ...")

    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);
    todos.data.mkdir("output")
    if not os.path.exists("output/image_facedet.torch"):
        model.save("output/image_facedet.torch")

    return model, device

def get_facegan_model():
    model = facegan.CodeFormer()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    if 'cpu' in str(device.type):
        model.float()

    print(f"Running on {device} ...")

    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);
    todos.data.mkdir("output")
    if not os.path.exists("output/image_facegan.torch"):
        model.save("output/image_facegan.torch")

    return model, device

def get_face_model():
    """Create model."""

    model = face.FaceModel()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    if 'cpu' in str(device.type):
        model.float()

    print(f"Running on {device} ...")

    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;                                                                                                             
    # # torch::jit::setTensorExprFuserEnabled(false);
    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_face.torch"):
    #     model.save("output/image_face.torch")

    return model, device


def face_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_face_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)

        with torch.no_grad():
            output_tensor, detected_tensor = model(input_tensor.to(device))


        output_file = f"{output_dir}/{os.path.basename(filename)}"
        todos.data.save_tensor([input_tensor, output_tensor], output_file)

        # save detected faces
        output_file = f"{output_dir}/d_{os.path.basename(filename)}"
        if detected_tensor.size(0) < 2:
            todos.data.save_tensor([detected_tensor], output_file)
        else:
            n_row = math.ceil(math.sqrt(detected_tensor.size(0)))
            if n_row % 2 != 0:
                n_row = n_row + 1
            grid_image = todos.data.grid_image(list(torch.split(detected_tensor, 1, dim=0)), nrow=n_row)
            grid_image.save(output_file)

    todos.model.reset_device()
