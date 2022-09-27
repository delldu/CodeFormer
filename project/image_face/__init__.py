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

import redos
import todos

from . import face

import pdb


def get_beauty_model():
    """Create beauty model."""

    model_path = "models/image_face.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = face.BeautyModel()
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_face.torch"):
        model.save("output/image_face.torch")

    return model, device


def beauty_model_forward(model, device, input_tensor, multi_times=1):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    torch.cuda.synchronize()
    with torch.jit.optimized_execution(False):
        output_tensor = todos.model.forward(model, device, input_tensor)
    torch.cuda.synchronize()

    return output_tensor[:, :, 0 : 2 * H, 0 : 2 * W]


def get_detect_model():
    """Create detect model."""

    model_path = "models/image_face.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = face.DetectModel()
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_face.torch"):
        model.save("output/image_face.torch")

    return model, device


def detect_model_forward(model, device, input_tensor, multi_times=1):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    torch.cuda.synchronize()
    with torch.jit.optimized_execution(False):
        output_tensor = todos.model.forward(model, device, input_tensor)
    torch.cuda.synchronize()

    return output_tensor


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.face_beautify(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, host="localhost", port=6379):
    # load model
    model, device = get_beauty_model()

    def do_service(input_file, output_file, targ):
        print(f"  face_beautify {input_file} ...")
        try:
            input_tensor = todos.data.load_rgba_tensor(input_file)
            output_tensor = beauty_model_forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_face", do_service, host, port)


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
        predict_tensor = beauty_model_forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        B, C, H, W = input_tensor.size()
        zoom2x_tensor = todos.data.resize_tensor(input_tensor, 2 * H, 2 * W)
        todos.data.save_tensor([zoom2x_tensor, predict_tensor], output_file)
    todos.model.reset_device()

def video_service(input_file, output_file, targ):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_beauty_model()

    print(f"  face_beautify {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def face_beautify_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)

        # # convert tensor from 1x4xHxW to 1x3xHxW
        # input_tensor = input_tensor[:, 0:3, :, :]
        output_tensor = beauty_model_forward(model, device, input_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=face_beautify_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.face_beautify(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, host="localhost", port=6379):
    return redos.video.service(name, "video_face_beautify", video_service, host, port)


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
        predict_tensor = detect_model_forward(model, device, input_tensor)
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
