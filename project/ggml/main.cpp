/************************************************************************************
***
*** Copyright 2021-2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, 2021年 11月 22日 星期一 14:33:18 CST
***
************************************************************************************/

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ggml_engine.h>
#include <nimage/tensor.h>

#include <facedet.h>

#define DEFAULT_DEVICE 1
#define DEFAULT_OUTPUT "output"


int image_face_client(RetinaFace *net, char *input_filename, char *output_filename)
{
    TENSOR *argv[1];

    printf("Face %s to %s ...\n", input_filename, output_filename);
    TENSOR *input_tensor = tensor_load_image(input_filename, 0 /*alpha*/);
    check_tensor(input_tensor);

    // // self.MAX_H = 2048
    // // self.MAX_W = 4096
    // // self.MAX_TIMES = 32
    // const int MAX_TIMES = 32;
    // int H = input_tensor->height;
    // int W = input_tensor->width;
    // int pad_h = (MAX_TIMES - (H % MAX_TIMES)) % MAX_TIMES;
    // int pad_w = (MAX_TIMES - (W % MAX_TIMES)) % MAX_TIMES;

    argv[0] = input_tensor ;
    TENSOR *output_tensor = net->engine_forward(ARRAY_SIZE(argv), argv);

    // TENSOR *xxxx_test = net->get_output_tensor("pad2d");
    // if (tensor_valid(xxxx_test)) {
    //     tensor_show("********************** pad2d", xxxx_test);
    //     tensor_destroy(xxxx_test);
    // }
    if (tensor_valid(output_tensor)) {
        // if (tensor_zeropad_(output_tensor, H, W) == RET_OK) {
        //     tensor_saveas_image(output_tensor, 0 /*batch*/, output_filename);
        // }
        tensor_show("output_tensor", output_tensor);

        tensor_destroy(output_tensor);
    }
    tensor_destroy(input_tensor);

    return RET_OK;
}


static void image_patch_help(char* cmd)
{
    printf("Usage: %s [option] image_files\n", cmd);
    printf("    -h, --help                   Display this help, version %s.\n", ENGINE_VERSION);
    printf("    -d, --device <no>            Set device (0 -- cpu, 1 -- cuda0, 2 -- cuda1, ..., default: %d)\n", DEFAULT_DEVICE);
    printf("    -o, --output                 output dir, default: %s.\n", DEFAULT_OUTPUT);

    exit(1);
}

int main(int argc, char** argv)
{
    int optc;
    int option_index = 0;
    int device_no = DEFAULT_DEVICE;
    char* output_dir = (char*)DEFAULT_OUTPUT;

    char *p, output_filename[1024];

    struct option long_opts[] = {
        { "help", 0, 0, 'h' },
        { "device", 1, 0, 'd' },
        { "output", 1, 0, 'o' },
        { 0, 0, 0, 0 }

    };

    if (argc <= 1)
        image_patch_help(argv[0]);


    while ((optc = getopt_long(argc, argv, "h d: o:", long_opts, &option_index)) != EOF) {
        switch (optc) {
        case 'd':
            device_no = atoi(optarg);
            break;
        case 'o':
            output_dir = optarg;
            break;
        case 'h': // help
        default:
            image_patch_help(argv[0]);
            break;
        }
    }

    // client
    if (optind == argc) // no input image, nothing to do ...
        return 0;

    RetinaFace net;

    // load net weight ...
    GGMLModel model;
    {
        check_point(model.preload("models/image_facedet_f32.gguf") == RET_OK);
        // model.remap("module.", "");

        // -----------------------------------------------------------------------------------------
        net.set_device(device_no);
        net.start_engine();
        net.dump();
    }

    for (int i = optind; i < argc; i++) {
        p = strrchr(argv[i], '/');
        if (p != NULL) {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, p + 1);
        } else {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, argv[i]);
        }

        net.load_weight(&model, "");
        image_face_client(&net, argv[i], output_filename);
    }

    // free network ...
    {
        model.clear();
        net.stop_engine();
    }

    return 0;
}
