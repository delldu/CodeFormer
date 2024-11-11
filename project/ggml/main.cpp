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
#include <facegan.h>

#define DEFAULT_DEVICE 1
#define DEFAULT_OUTPUT "output"

// -----------------------------------------------------------------------------------------
TENSOR *facedet_forward(RetinaFace *net, TENSOR *input_tensor);
TENSOR *facegan_forward(CodeFormer *net, TENSOR *input_tensor);
int face_gan(CodeFormer *net, TENSOR* input, TENSOR* detect_result);

int face_detect(RetinaFace *net, char *input_file, char *output_file);
int face_beauty(RetinaFace *det_net, CodeFormer *gan_net, char *input_file, char *output_file);
int image_face_detect(int device, int argc, char** argv, char *output_dir);
int image_face_beauty(int device, int argc, char** argv, char *output_dir);
// -----------------------------------------------------------------------------------------

static void image_face_help(char* cmd)
{
    printf("Usage: %s [option] image_files\n", cmd);
    printf("    -h, --help                   Display this help, version %s.\n", ENGINE_VERSION);
    printf("    -d, --device <no>            Set device (0 -- cpu, 1 -- cuda0, 2 -- cuda1, ..., default: %d)\n", DEFAULT_DEVICE);
    printf("    -m, --mode <no>              Set mode (0 -- detect, 1 -- beauty, default: 1 -- beauty)\n");
    printf("    -o, --output                 output dir, default: %s.\n", DEFAULT_OUTPUT);

    exit(1);
}

int main(int argc, char** argv)
{
    int optc;
    int option_index = 0;
    int device = DEFAULT_DEVICE;
    int mode = 1;
    char* output_dir = (char*)DEFAULT_OUTPUT;

    char *p, output_filename[1024];

    struct option long_opts[] = {
        { "help", 0, 0, 'h' },
        { "device", 1, 0, 'd' },
        { "mode", 1, 0, 'm' },
        { "output", 1, 0, 'o' },
        { 0, 0, 0, 0 }

    };

    if (argc <= 1)
        image_face_help(argv[0]);


    while ((optc = getopt_long(argc, argv, "h d: m: o:", long_opts, &option_index)) != EOF) {
        switch (optc) {
        case 'd':
            device = atoi(optarg);
            break;
        case 'm':
            mode = atoi(optarg);
            break;
        case 'o':
            output_dir = optarg;
            break;
        case 'h': // help
        default:
            image_face_help(argv[0]);
            break;
        }
    }

    // client
    if (optind == argc) // no input image, nothing to do ...
        return 0;

    make_dir(output_dir);
    if (mode == 0) {
        return image_face_detect(device, argc - optind, &argv[optind], output_dir);
    }

    return image_face_beauty(device, argc - optind, &argv[optind], output_dir);
}
