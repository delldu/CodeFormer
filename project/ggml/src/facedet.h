#ifndef __FACEDET__H__
#define __FACEDET__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#include <vector>

#pragma GCC diagnostic ignored "-Wformat-truncation"

// ggml_set_name(out, "xxxx_test");
// ggml_set_output(out);


/*
 LandmarkHead(
  (conv1x1): Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
) */

struct LandmarkHead {
    // network hparams
    int in_channels = 256;
    int num_anchors = 2;

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
        conv1x1.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
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
    int in_channels = 256;
    int num_anchors = 2;

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
        conv1x1.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
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
    int in_channels = 256;
    int num_anchors = 2;

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
        conv1x1.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
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
 
        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "1.");
        bn.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = bn.forward(ctx, x);

        return x;
    }
};


struct SSH {
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

        snprintf(s, sizeof(s), "%s%s", prefix, "conv7x7_3.");
        conv7X7_3.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *conv3 = conv3X3.forward(ctx, x);

        x = conv5X5_1.forward(ctx, x);
        ggml_tensor_t *conv5X5_1 = ggml_leaky_relu(ctx, x, 0.0, false /*inplace*/);
        ggml_tensor_t *conv5 = conv5X5_2.forward(ctx, conv5X5_1);
        
        conv5X5_1 = conv7X7_2.forward(ctx, conv5X5_1);
        ggml_tensor_t *conv7X7_2 = ggml_leaky_relu(ctx, conv5X5_1, 0.0, false /*inplace*/);
        ggml_tensor_t *conv7 = conv7X7_3.forward(ctx, conv7X7_2);

        ggml_tensor_t *out = ggml_concat(ctx, conv3, conv5, 2/*dim*/);
        out = ggml_concat(ctx, out, conv7, 2/*dim*/);
    	return ggml_relu(ctx, out);
    }
};

struct FpnLayer {
    int in_channels = 256;
    int out_channels = 256;
    int kernel_size = 1;
    int padding_size = 0;

    struct Conv2d conv;
    struct BatchNorm2d bn;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = in_channels;
        conv.out_channels = out_channels;
        conv.kernel_size = { kernel_size, kernel_size };
        conv.padding = { padding_size, padding_size };
        conv.stride = { 1, 1 };
        conv.has_bias = false;
        conv.create_weight_tensors(ctx);

        bn.num_features = out_channels;
        bn.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "1.");
        bn.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = bn.forward(ctx, x);
        x = ggml_leaky_relu(ctx, x, 0.0, true);

        return x;
    }
};


struct FPN {
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
        merge1.kernel_size = 3;
        merge1.padding_size = 1;
        merge1.create_weight_tensors(ctx);

        merge2.in_channels = 256;
        merge2.out_channels = 256;
        merge2.kernel_size = 3;
        merge2.padding_size = 1;
        merge2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
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

    std::vector<ggml_tensor_t*> forward(struct ggml_context* ctx, ggml_tensor_t* x1, ggml_tensor_t* x2, ggml_tensor_t* x3) {
        std::vector<ggml_tensor_t*> fpn_out;

        ggml_tensor_t *y1 = output1.forward(ctx, x1);
        ggml_tensor_t *y2 = output2.forward(ctx, x2);
        ggml_tensor_t *y3 = output3.forward(ctx, x3);

        // Update y2
        ggml_tensor_t *up3 = ggml_upscale_ext(ctx, y3, y2->ne[0], y2->ne[1], y3->ne[2], y3->ne[3]); // W, H, C, B
        y2 = ggml_add(ctx, y2, up3);
        y2 = merge2.forward(ctx, y2);

        // Update y1
        ggml_tensor_t *up2 = ggml_upscale_ext(ctx, y2, y1->ne[0], y1->ne[1], y2->ne[2], y2->ne[3]); // W, H, C, B
        y1 = ggml_add(ctx, y1, up2);
        y1 = merge1.forward(ctx, y1);

        fpn_out.push_back(y1);
        fpn_out.push_back(y2);
        fpn_out.push_back(y3);

    	return fpn_out;
    }
};

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
 
        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "1.");
        bn.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = bn.forward(ctx, x);

        return x;
    }
};


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

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, struct BottleneckDownsample *down) {
        ggml_tensor_t *id;
        if (down) {
            id = down->forward(ctx, x);
        } else {
            id = x;
        }

        x = conv1.forward(ctx, x);
        x = bn1.forward(ctx, x);
        x = ggml_relu(ctx, x);

        x = conv2.forward(ctx, x);
        x = bn2.forward(ctx, x);
        x = ggml_relu(ctx, x);

        x = conv3.forward(ctx, x);
        x = bn3.forward(ctx, x);

        x = ggml_add(ctx, x, id);
        x = ggml_relu(ctx, x);
    	return x;
    }
};

struct ResNet3Layers {
    // network hparams
    // int in_planes = 2048;

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

        // maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        maxpool.kernel_size = 3;
        maxpool.stride = 2;
        maxpool.padding = 1;
        maxpool.create_weight_tensors(ctx);

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

    void setup_weight_names(const char *prefix) {
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

    std::vector<ggml_tensor_t*> forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        std::vector<ggml_tensor_t*> resnet3_out;

        x = conv1.forward(ctx, x);
        x = bn1.forward(ctx, x);
        x = ggml_relu(ctx, x);
        x = maxpool.forward(ctx, x);

        // layer1
        x = layer1_0.forward(ctx, x, &layer1_0_downsample);
        x = layer1_1.forward(ctx, x, NULL);
        ggml_tensor_t *x1 = layer1_2.forward(ctx, x, NULL);

        // layer2
        x1 = layer2_0.forward(ctx, x1, &layer2_0_downsample);
        x1 = layer2_1.forward(ctx, x1, NULL);
        x1 = layer2_2.forward(ctx, x1, NULL);
        ggml_tensor_t *x2 = layer2_3.forward(ctx, x1, NULL);
        resnet3_out.push_back(x2);

        // layer3
        x2 = layer3_0.forward(ctx, x2, &layer3_0_downsample);
        x2 = layer3_1.forward(ctx, x2, NULL);
        x2 = layer3_2.forward(ctx, x2, NULL);
        x2 = layer3_3.forward(ctx, x2, NULL);
        x2 = layer3_4.forward(ctx, x2, NULL);
        ggml_tensor_t *x3 = layer3_5.forward(ctx, x2, NULL);
        resnet3_out.push_back(x3);

        // layer4
        x3 = layer4_0.forward(ctx, x3, &layer4_0_downsample);
        x3 = layer4_1.forward(ctx, x3, NULL);
        ggml_tensor_t *x4 = layer4_2.forward(ctx, x3, NULL);
        resnet3_out.push_back(x4);

    	return resnet3_out;
    }
};

struct RetinaFace : GGMLNetwork {
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

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "module.body.");
        body.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "module.fpn.");
        fpn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "module.ssh1.");
        ssh1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "module.ssh2.");
        ssh2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "module.ssh3.");
        ssh3.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "module.ClassHead.0.");
        ClassHead_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "module.ClassHead.1.");
        ClassHead_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "module.ClassHead.2.");
        ClassHead_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "module.BboxHead.0.");
        BboxHead_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "module.BboxHead.1.");
        BboxHead_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "module.BboxHead.2.");
        BboxHead_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "module.LandmarkHead.0.");
        LandmarkHead_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "module.LandmarkHead.1.");
        LandmarkHead_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "module.LandmarkHead.2.");
        LandmarkHead_2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_UNUSED(argc);

        ggml_tensor_t *x = argv[0];

        // # tensor [x] size: [1, 3, 351, 500], min: -108.986504, max: 126.011002, mean: -26.315453
        std::vector<ggml_tensor_t *> resnet3_out = body.forward(ctx, x); // x -- bgr image
        ggml_tensor_t* x1 = resnet3_out[0];
        ggml_tensor_t* x2 = resnet3_out[1];
        ggml_tensor_t* x3 = resnet3_out[2];

        // # tensor [bgr_image] size: [1, 3, 351, 500], min: -108.986504, max: 126.011002, mean: -26.315453
        // # out is list: len = 3
        // #     tensor [item] size: [1, 512, 44, 63], min: 0.0, max: 2.788131, mean: 0.08262
        // #     tensor [item] size: [1, 1024, 22, 32], min: 0.0, max: 2.534333, mean: 0.033747
        // #     tensor [item] size: [1, 2048, 11, 16], min: 0.0, max: 6.305652, mean: 0.326206
        std::vector<ggml_tensor_t *> fpn_out = fpn.forward(ctx, x1, x2, x3);
        // # fpn is list: len = 3
        // #     tensor [item] size: [1, 256, 44, 63], min: -0.0, max: 6.100448, mean: 0.307941
        // #     tensor [item] size: [1, 256, 22, 32], min: -0.0, max: 8.648984, mean: 0.271562
        // #     tensor [item] size: [1, 256, 11, 16], min: -0.0, max: 8.366714, mean: 0.319153

        ggml_tensor_t *f0 = ssh1.forward(ctx, fpn_out[0]);
        ggml_tensor_t *f1 = ssh2.forward(ctx, fpn_out[1]);
        ggml_tensor_t *f2 = ssh3.forward(ctx, fpn_out[2]);
        // # tensor [feature1] size: [1, 256, 44, 63], min: 0.0, max: 5.633877, mean: 0.342905
        // # tensor [feature2] size: [1, 256, 22, 32], min: 0.0, max: 5.345377, mean: 0.321881
        // # tensor [feature3] size: [1, 256, 11, 16], min: 0.0, max: 3.646523, mean: 0.264832

        // BBox regressions ...
        ggml_tensor_t *box0 = BboxHead_0.forward(ctx, f0);
        ggml_tensor_t *box1 = BboxHead_1.forward(ctx, f1);
        ggml_tensor_t *box2 = BboxHead_2.forward(ctx, f2);
        ggml_tensor_t *bbox_regressions = ggml_concat(ctx, box0, box1, 1/*dim*/);
        bbox_regressions = ggml_concat(ctx, bbox_regressions, box2, 1/*dim*/);
        // # tensor [bbox_regressions] size: [1, 7304, 4], min: -4.950671, max: 5.339179, mean: -0.02128

        // Score regressions ...
        ggml_tensor_t *score0 = ClassHead_0.forward(ctx, f0);
        ggml_tensor_t *score1 = ClassHead_1.forward(ctx, f1);
        ggml_tensor_t *score2 = ClassHead_2.forward(ctx, f2);
        ggml_tensor_t *score_regressions = ggml_concat(ctx, score0, score1, 1/*dim*/);
        score_regressions = ggml_concat(ctx, score_regressions, score2, 1/*dim*/); // f32 [2, 7304, 1, 1]

        score_regressions = ggml_soft_max(ctx, score_regressions);

        // # tensor [classifications] size: [1, 7304, 2], min: -7.161493, max: 6.567798, mean: -0.06647

        // Landmark regressions ...
        ggml_tensor_t *ldm0 = LandmarkHead_0.forward(ctx, f0);
        ggml_tensor_t *ldm1 = LandmarkHead_1.forward(ctx, f1);
        ggml_tensor_t *ldm2 = LandmarkHead_2.forward(ctx, f2);
        ggml_tensor_t *ldm_regressions = ggml_concat(ctx, ldm0, ldm1, 1/*dim*/);
        ldm_regressions = ggml_concat(ctx, ldm_regressions, ldm2, 1/*dim*/);
        // # tensor [ldm_regressions] size: [1, 7304, 10], min: -10.258643, max: 11.092538, mean: 0.105917

        // bbox_regressions    f32 [4, 7304, 1, 1], 
        // score_regressions    f32 [2, 7304, 1, 1], 
        // ldm_regressions    f32 [10, 7304, 1, 1], 

        ggml_tensor_t *conf_loc_landmarks = ggml_concat(ctx, score_regressions, bbox_regressions, 0/*dim*/);
        conf_loc_landmarks = ggml_concat(ctx, conf_loc_landmarks, ldm_regressions, 0/*dim*/);
        // # tensor [conf_loc_landmarks] size: [7304, 16], min: -10.258643, max: 11.092538, mean: 0.123378
        ggml_set_name(conf_loc_landmarks, "conf_loc_landmarks");
        ggml_set_output(conf_loc_landmarks);

        // Info: output_tensor Tensor: 1x1x7304x16
        // min: -10.2714, max: 11.0866, mean: 0.1235

    	return conf_loc_landmarks;
    }
};

#endif // __FACEDET__H__
