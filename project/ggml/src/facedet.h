#ifndef __FACEDET__H__
#define __FACEDET__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

/*
 LandmarkHead(
  (conv1x1): Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
) */

struct LandmarkHead {
    // network hparams
    int in_channels == 256;
    int num_anchors == 2

    // network params
    struct Conv2d conv1x1;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1x1.in_channels = in_channels;
        conv1x1.out_channels = num_anchors * 10;
        conv1x1.kernel_size = {1, 1};

        conv1x1.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1x1.");
        conv_layer.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        x = conv1x1.forward(ctx, x);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); // [63, 44, 20, 1] --> [20, 63, 44, 1]
        int64_t n12 = x->ne[1]*x->ne[2];
        int64_t n3 = x->ne[3];
        return ggml_reshape_3d(ctx, x, 10, 2*n12, n3);
    }
};

/*
 BboxHead(
  (conv1x1): Conv2d(256, 8, kernel_size=(1, 1), stride=(1, 1))
) */

struct BboxHead {
    // network hparams
    int in_channels == 256;
    int num_anchors == 2

    // network params
    struct Conv2d conv1x1;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1x1.in_channels = in_channels;
        conv1x1.out_channels = num_anchors * 4;
        conv1x1.kernel_size = {1, 1};

        conv1x1.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1x1.");
        conv_layer.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        x = conv1x1.forward(ctx, x);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); // [63, 44, 8, 1] --> [8, 63, 44, 1]
        int64_t n12 = x->ne[1]*x->ne[2];
        int64_t n3 = x->ne[3];
        return ggml_reshape_3d(ctx, x, 4, 2*n12, n3);
    }
};

/*
 ClassHead(
  (conv1x1): Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
) */
struct ClassHead {
    // network hparams
    int in_channels == 256;
    int num_anchors == 2

    // network params
    struct Conv2d conv1x1;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1x1.in_channels = in_channels;
        conv1x1.out_channels = num_anchors * 2;
        conv1x1.kernel_size = {1, 1};

        conv1x1.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1x1.");
        conv_layer.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        x = conv1x1.forward(ctx, x);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); // [63, 44, 4, 1] --> [4, 63, 44, 1]
        int64_t n12 = x->ne[1]*x->ne[2];
        int64_t n3 = x->ne[3];
        return ggml_reshape_3d(ctx, x, 2, 2*n12, n3);
    }
};

struct SshLayer {
    int in_channels = 256;
    int out_channels = 128;

    struct Conv2d conv;
    struct BatchNorm2d bn;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = in_channels;
        conv.out_channels = out_channels;
        conv.kernel_size = { 3, 3 };
        conv.stride = { 1, 1 };
        conv.padding = { 1, 1 };
        conv.has_bias = false;
        conv.create_weight_tensors(ctx);

        bn.num_features = out_channels;
        bn.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn.");
        bn.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = bn.forward(ctx, x);

        return x;
    }
};


struct SSH {
    // network hparams

    struct SshLayer conv3X3;
    struct SshLayer conv5X5_1;
    struct SshLayer conv5X5_2;
    struct SshLayer conv7X7_2;
    struct SshLayer conv7X7_3;


    void create_weight_tensors(struct ggml_context* ctx) {
        conv3X3.in_channels = 256;
        conv3X3.out_channels = 128;
        conv3X3.create_weight_tensors(ctx);

        conv5X5_1.in_channels = 256;
        conv5X5_1.out_channels = 64;
        conv5X5_1.create_weight_tensors(ctx);

        conv5X5_2.in_channels = 64;
        conv5X5_2.out_channels = 64;
        conv5X5_2.create_weight_tensors(ctx);

        conv7X7_2.in_channels = 64;
        conv7X7_2.out_channels = 64;
        conv7X7_2.create_weight_tensors(ctx);

        conv7X7_3.in_channels = 64;
        conv7X7_3.out_channels = 64;
        conv7X7_3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "conv3X3.");
        conv3X3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv5X5_1.");
        conv5X5_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv5X5_2.");
        conv5X5_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv7X7_2.");
        conv7X7_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv7X7_3.");
        conv7X7_3.setup_weight_names(s);
    }

    // GGML_API struct ggml_tensor * ggml_leaky_relu(
    //         struct ggml_context * ctx,
    //         struct ggml_tensor  * a, float negative_slope, bool inplace);
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        x = conv3X3.forward(ctx, x);

        x = conv5X5_1.forward(ctx, x);
        x = ggml_leaky_relu(ctx, x, 0.0, true /*inplace*/);
        x = conv5X5_2.forward(ctx, x);
        
        x = conv7X7_2.forward(ctx, x);
        x = ggml_leaky_relu(ctx, x, 0.0, true /*inplace*/);
        x = conv7X7_3.forward(ctx, x);

    	return x;
    }
};



struct FpnLayer {
    int in_channels = 256;
    int out_channels = 256;

    struct Conv2d conv;
    struct BatchNorm2d bn;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = in_channels;
        conv.out_channels = out_channels;
        conv.kernel_size = { 1, 1 };
        conv.stride = { 1, 1 };
        conv.has_bias = false;
        conv.create_weight_tensors(ctx);

        bn.num_features = out_channels;
        bn.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn.");
        bn.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = bn.forward(ctx, x);

        return x;
    }
};

struct FPN {
    // network hparams

    // network params
    struct FpnLayer output1;
    struct FpnLayer output2;
    struct FpnLayer output3;

    struct FpnLayer merge1;
    struct FpnLayer merge2;

    void create_weight_tensors(struct ggml_context* ctx) {
        // in_channels_list:  [512, 1024, 2048]
        // out_channels:  256
        output1.in_channels = 512;
        output1.out_channels = 256;
        output1.create_weight_tensors(ctx);

        output2.in_channels = 1024;
        output2.out_channels = 256;
        output2.create_weight_tensors(ctx);

        output3.in_channels = 2048;
        output3.out_channels = 256;
        output3.create_weight_tensors(ctx);

        merge1.in_channels = 256;
        merge1.out_channels = 256;
        merge1.create_weight_tensors(ctx);

        merge2.in_channels = 256;
        merge2.out_channels = 256;
        merge2.create_weight_tensors(ctx);
    }

    void setup_weight_names(char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "output1.");
        output1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "output2.");
        output2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "output3.");
        output3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "merge1.");
        merge1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "merge2.");
        merge2.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!
        x = output1.forward(ctx, x);
        x = ggml_leaky_relu(ctx, x, 0.0, true /*inplace*/);


    	return x;
    }
};

/*
 Bottleneck(
  (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
) */

struct Bottleneck {
    // network hparams
    int in_planes;
    int out_planes;
    int stride;

    struct Conv2d conv1;
    struct BatchNorm2d bn1;
    struct Conv2d conv2;
    struct BatchNorm2d bn2;
    struct Conv2d conv3;
    struct BatchNorm2d bn3;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = in_planes;
        conv1.out_channels = out_planes;
        conv1.kernel_size = { 1, 1 };
        conv1.stride = { 1, 1 };
        conv1.has_bias = false;
        conv1.create_weight_tensors(ctx);

        bn1.num_features = out_planes;
        bn1.create_weight_tensors(ctx);

        conv2.in_channels = out_planes;
        conv2.out_channels = out_planes;
        conv2.kernel_size = { 3, 3 };
        conv2.stride = { stride, stride };
        conv2.padding = { 1, 1 };
        conv2.has_bias = false;
        conv2.create_weight_tensors(ctx);

        bn2.num_features = out_planes;
        bn2.create_weight_tensors(ctx);

        conv3.in_channels = out_planes;
        conv3.out_channels = out_planes * 4;
        conv3.kernel_size = { 1, 1 };
        conv3.stride = { 1, 1 };
        conv3.has_bias = false;
        conv3.create_weight_tensors(ctx);

        bn3.num_features = out_planes * 4;
        bn3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "bn1.");
        bn1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "bn2.");
        bn2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv3.");
        conv3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "bn3.");
        bn3.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

// downsample = nn.Sequential(
//     conv1x1(self.in_planes, out_planes * block.expansion, stride),
//     nn.BatchNorm2d(out_planes * block.expansion),
// )

struct BottleneckDownsample {
    int in_planes;
    int out_planes;
    int stride;

    struct Conv2d conv;
    struct BatchNorm2d bn;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = in_planes;
        conv.out_channels = out_planes * 4;
        conv.kernel_size = { 1, 1 };
        conv.stride = { stride, stride };
        conv.has_bias = false;
        conv.create_weight_tensors(ctx);

        bn.num_features = out_planes * 4;
        bn.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn.");
        bn.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = bn.forward(ctx, x);

        return x;
    }
};

struct ResNet3Layers {
    // network hparams
    int in_planes = 2048;

    // network params

    struct Conv2d conv1;
    struct BatchNorm2d bn1;

    struct MaxPool2d maxpool;

    struct Bottleneck layer1_0;
    struct BottleneckDownsample layer1_0_downsample;
    struct Bottleneck layer1_1;
    struct Bottleneck layer1_2;

    struct Bottleneck layer2_0;
    struct BottleneckDownsample layer2_0_downsample;
    struct Bottleneck layer2_1;
    struct Bottleneck layer2_2;
    struct Bottleneck layer2_3;

    struct Bottleneck layer3_0;
    struct BottleneckDownsample layer3_0_downsample;
    struct Bottleneck layer3_1;
    struct Bottleneck layer3_2;
    struct Bottleneck layer3_3;
    struct Bottleneck layer3_4;
    struct Bottleneck layer3_5;

    struct Bottleneck layer4_0;
    struct BottleneckDownsample layer4_0_downsample;
    struct Bottleneck layer4_1;
    struct Bottleneck layer4_2;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = 3;
        conv1.out_channels = 64;
        conv1.kernel_size = { 7, 7 };
        conv1.stride = { 2, 2 };
        conv1.padding = { 3, 3 };
        conv1.has_bias = false;
        conv1.create_weight_tensors(ctx);

        bn1.num_features = 64;
        bn1.create_weight_tensors(ctx);

        // layer1
        layer1_0.in_planes = 64;
        layer1_0.out_planes = 64;
        layer1_0.stride = 1;
        layer1_0.create_weight_tensors(ctx);

        layer1_0_downsample.in_planes = 64;
        layer1_0_downsample.out_planes = 64;
        layer1_0_downsample.stride = 1;
        layer1_0_downsample.create_weight_tensors(ctx);

        layer1_1.in_planes = 256;
        layer1_1.out_planes = 64;
        layer1_1.stride = 1;
        layer1_1.create_weight_tensors(ctx);

        layer1_2.in_planes = 256;
        layer1_2.out_planes = 64;
        layer1_2.stride = 1;
        layer1_2.create_weight_tensors(ctx);

        // layer2
        layer2_0.in_planes = 256;
        layer2_0.out_planes = 128;
        layer2_0.stride = 2;
        layer2_0.create_weight_tensors(ctx);

        layer2_0_downsample.in_planes = 256;
        layer2_0_downsample.out_planes = 128;
        layer2_0_downsample.stride = 2;
        layer2_0_downsample.create_weight_tensors(ctx);

        layer2_1.in_planes = 512;
        layer2_1.out_planes = 128;
        layer2_1.stride = 1;
        layer2_1.create_weight_tensors(ctx);

        layer2_2.in_planes = 512;
        layer2_2.out_planes = 128;
        layer2_2.stride = 1;
        layer2_2.create_weight_tensors(ctx);

        layer2_3.in_planes = 512;
        layer2_3.out_planes = 128;
        layer2_3.stride = 1;
        layer2_3.create_weight_tensors(ctx);

        // layer3
        layer3_0.in_planes = 512;
        layer3_0.out_planes = 256;
        layer3_0.stride = 2;
        layer3_0.create_weight_tensors(ctx);

        layer3_0_downsample.in_planes = 512;
        layer3_0_downsample.out_planes = 256;
        layer3_0_downsample.stride = 2;
        layer3_0_downsample.create_weight_tensors(ctx);

        layer3_1.in_planes = 1024;
        layer3_1.out_planes = 256;
        layer3_1.stride = 1;
        layer3_1.create_weight_tensors(ctx);

        layer3_2.in_planes = 1024;
        layer3_2.out_planes = 256;
        layer3_2.stride = 1;
        layer3_2.create_weight_tensors(ctx);

        layer3_3.in_planes = 1024;
        layer3_3.out_planes = 256;
        layer3_3.stride = 1;
        layer3_3.create_weight_tensors(ctx);

        layer3_4.in_planes = 1024;
        layer3_4.out_planes = 256;
        layer3_4.stride = 1;
        layer3_4.create_weight_tensors(ctx);

        layer3_5.in_planes = 1024;
        layer3_5.out_planes = 256;
        layer3_5.stride = 1;
        layer3_5.create_weight_tensors(ctx);

        // layer4
        layer4_0.in_planes = 1024;
        layer4_0.out_planes = 512;
        layer4_0.stride = 2;
        layer4_0.create_weight_tensors(ctx);

        layer4_0_downsample.in_planes = 1024;
        layer4_0_downsample.out_planes = 512;
        layer4_0_downsample.stride = 2;
        layer4_0_downsample.create_weight_tensors(ctx);

        layer4_1.in_planes = 2048;
        layer4_1.out_planes = 512;
        layer4_1.stride = 1;
        layer4_1.create_weight_tensors(ctx);

        layer4_2.in_planes = 2048;
        layer4_2.out_planes = 512;
        layer4_2.stride = 1;
        layer4_2.create_weight_tensors(ctx);
    }

    void setup_weight_names(char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "bn1.");
        bn1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layer1.0.");
        layer1_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer1.0.downsample.");
        layer1_0_downsample.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer1.1.");
        layer1_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer1.2.");
        layer1_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.0.");
        layer2_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.0.downsample.");
        layer2_0_downsample.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.1.");
        layer2_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.2.");
        layer2_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.3.");
        layer2_3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.0.");
        layer3_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.0.downsample.");
        layer3_0_downsample.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.1.");
        layer3_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.2.");
        layer3_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.3.");
        layer3_3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.4.");
        layer3_4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.5.");
        layer3_5.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layer4.0.");
        layer4_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer4.0.downsample.");
        layer4_0_downsample.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer4.1.");
        layer4_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer4.2.");
        layer4_2.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!


    	return x;
    }
};


struct RetinaFace {
    // network params
    struct ResNet3Layers body;

    struct FPN fpn;

    struct SSH ssh1;
    struct SSH ssh2;
    struct SSH ssh3;

    struct ClassHead ClassHead_0;
    struct ClassHead ClassHead_1;
    struct ClassHead ClassHead_2;

    struct BboxHead BboxHead_0;
    struct BboxHead BboxHead_1;
    struct BboxHead BboxHead_2;

    struct LandmarkHead LandmarkHead_0;
    struct LandmarkHead LandmarkHead_1;
    struct LandmarkHead LandmarkHead_2;


    void create_weight_tensors(struct ggml_context* ctx) {
        body.create_weight_tensors(ctx);

        fpn.create_weight_tensors(ctx);

        ssh1.create_weight_tensors(ctx);
        ssh2.create_weight_tensors(ctx);
        ssh3.create_weight_tensors(ctx);

        ClassHead_0.create_weight_tensors(ctx);
        ClassHead_1.create_weight_tensors(ctx);
        ClassHead_2.create_weight_tensors(ctx);

        BboxHead_0.create_weight_tensors(ctx);
        BboxHead_1.create_weight_tensors(ctx);
        BboxHead_2.create_weight_tensors(ctx);

        LandmarkHead_0.create_weight_tensors(ctx);
        LandmarkHead_1.create_weight_tensors(ctx);
        LandmarkHead_2.create_weight_tensors(ctx);
    }

    void setup_weight_names(char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "body.");
        body.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "ssh1.");
        ssh1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ssh2.");
        ssh2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ssh3.");
        ssh3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "ClassHead.0.");
        ClassHead_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ClassHead.1.");
        ClassHead_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ClassHead.2.");
        ClassHead_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "BboxHead.0.");
        BboxHead_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "BboxHead.1.");
        BboxHead_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "BboxHead.2.");
        BboxHead_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "LandmarkHead.0.");
        LandmarkHead_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "LandmarkHead.1.");
        LandmarkHead_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "LandmarkHead.2.");
        LandmarkHead_2.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {

    	return x;
    }
};

#endif // __FACEDET__H__
