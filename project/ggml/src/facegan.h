#ifndef __FACEGAN__H__
#define __FACEGAN__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"


/*
 MultiheadAttention(
  (out_proj): Linear(in_features=512, out_features=512, bias=True)
) */

struct MultiheadAttention {
    // network hparams
    int num_heads = 8;
    int head_dim = 64;

    // network params
    struct Linear out_proj;


    void create_weight_tensors(struct ggml_context* ctx) {
        out_proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[512];
        snprintf(s, sizeof(s), "%s%s", prefix, "out_proj.");
        out_proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!
      // xxxx_debug
    	return x;
    }
};

/*
 TransformerSALayer(
  (self_attn): MultiheadAttention(
    (out_proj): Linear(in_features=512, out_features=512, bias=True)
  )
  (linear1): Linear(in_features=512, out_features=1024, bias=True)
  (linear2): Linear(in_features=1024, out_features=512, bias=True)
  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
) */

struct TransformerSALayer {
    // network hparams

    // network params
    struct MultiheadAttention self_attn; // xxxx_debug

    struct Linear linear1;
    struct Linear linear2;

    struct LayerNorm norm1;
    struct LayerNorm norm2;

    void create_weight_tensors(struct ggml_context* ctx) {
        linear1.in_features = 512;
        linear1.out_features = 1024;
        linear1.create_weight_tensors(ctx);

        linear2.in_features = 1024;
        linear2.out_features = 512;
        linear2.create_weight_tensors(ctx);

        norm1.normalized_shape = 512;
        norm1.create_weight_tensors(ctx);

        norm2.normalized_shape = 512;
        norm2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "linear1.");
        linear1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "linear2.");
        linear2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!
      // xxxx_debug
    	return x;
    }
};

struct Upsample {
    int in_channels;

    // network params
    struct Conv2d conv;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        conv.in_channels = in_channels;
        conv.out_channels = in_channels;
        conv.kernel_size = {3, 3};
        conv.stride = { 1, 1 };
        conv.padding = { 1, 1 };
        // conv.has_bias = true;
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
      // x = F.interpolate(x, scale_factor=2.0, mode="nearest")
      // x = self.conv(x)
      x = ggml_upscale(ctx, x, 2);
      x = conv.forward(ctx, x);

    	return x;
    }
};

struct ResBlock {
    int in_channels;
    int out_channels;

    // network params
    struct GroupNorm norm1;
    struct Conv2d conv1;
    struct GroupNorm norm2;
    struct Conv2d conv2;

    struct Conv2d conv_out;

    void create_weight_tensors(struct ggml_context* ctx) {
        norm1.num_channels = in_channels;
        norm1.create_weight_tensors(ctx);

        conv1.in_channels = in_channels;
        conv1.out_channels = out_channels;
        conv1.kernel_size = { 3, 3 };
        conv1.stride = { 1, 1 };
        conv1.padding = { 1, 1 };
        conv1.create_weight_tensors(ctx);

        norm2.num_channels = out_channels;
        norm2.create_weight_tensors(ctx);

        conv2.in_channels = out_channels;
        conv2.out_channels = out_channels;
        conv2.kernel_size = { 3, 3 };
        conv2.stride = { 1, 1 };
        conv2.padding = { 1, 1 };
        // conv2.has_bias = false;
        conv2.create_weight_tensors(ctx);

        if (in_channels != out_channels) {
          conv_out.in_channels = in_channels;
          conv_out.out_channels = out_channels;
          conv_out.kernel_size = { 1, 1 };
          conv_out.stride = { 1, 1 };
          conv_out.padding = { 0, 0 };
          // conv_out.has_bias = false;
          conv_out.create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);

        if (in_channels != out_channels) {
          snprintf(s, sizeof(s), "%s%s", prefix, "conv_out.");
          conv_out.setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x = x_in
        // x = self.norm1(x)
        // x = swish(x)
        // x = self.conv1(x)
        // x = self.norm2(x)
        // x = swish(x)
        // x = self.conv2(x)

        // x_in = self.conv_out(x_in)

        // return x + x_in
        ggml_tensor_t* s;
        if (in_channels != out_channels) {
          s = conv_out.forward(ctx, x);
        } else {
          s = x;
        }
        
        x = norm1.forward(ctx, x);
        // x = x * torch.sigmoid(x)
        x = ggml_mul(ctx, x, ggml_sigmoid(ctx, x));
        x = conv1.forward(ctx, x);
        x = norm2.forward(ctx, x);
        // x = swish(x)
        x = ggml_mul(ctx, x, ggml_sigmoid(ctx, x));
        x = conv2.forward(ctx, x);

        x = ggml_add(ctx, x, s);

        return x;
    }
};

/*
 AttnBlock(
  (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
  (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
  (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
  (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
  (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
) */

struct AttnBlock {
    int in_channels = 512;
    // int out_channels;
    
    struct GroupNorm norm;
    struct Conv2d q;
    struct Conv2d k;
    struct Conv2d v;
    struct Conv2d proj_out;

    void create_weight_tensors(struct ggml_context* ctx) {
        norm.num_channels = in_channels;
        norm.eps = 1e-6;
        norm.create_weight_tensors(ctx);

        q.in_channels = in_channels;
        q.out_channels = in_channels;
        q.kernel_size = { 1, 1 };
        q.stride = { 1, 1 };
        q.padding = { 0, 0 };
        // q.has_bias = false;
        q.create_weight_tensors(ctx);

        k.in_channels = in_channels;
        k.out_channels = in_channels;
        k.kernel_size = { 1, 1 };
        k.stride = { 1, 1 };
        k.padding = { 0, 0 };
        // k.has_bias = false;
        k.create_weight_tensors(ctx);

        v.in_channels = in_channels;
        v.out_channels = in_channels;
        v.kernel_size = { 1, 1 };
        v.stride = { 1, 1 };
        v.padding = { 0, 0 };
        // k.has_bias = false;
        v.create_weight_tensors(ctx);

        proj_out.in_channels = in_channels;
        proj_out.out_channels = in_channels;
        proj_out.kernel_size = { 1, 1 };
        proj_out.stride = { 1, 1 };
        proj_out.padding = { 0, 0 };
        // proj_out.has_bias = false;
        proj_out.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "q.");
        q.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "k.");
        k.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "v.");
        v.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "proj_out.");
        proj_out.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
      // please implement forward by your self, please !!!
      // xxxx_debug
      return x;
    }
};

struct Generator {
    // network hparams
    // int num_resolutions = 6;

    // network params
    struct Conv2d blocks_0;
    struct ResBlock blocks_1;
    struct AttnBlock blocks_2;
    struct ResBlock blocks_3;
    struct ResBlock blocks_4;
    struct AttnBlock blocks_5;
    struct ResBlock blocks_6;
    struct AttnBlock blocks_7;
    struct Upsample blocks_8;
    struct ResBlock blocks_9;
    struct ResBlock blocks_10;
    struct Upsample blocks_11;
    struct ResBlock blocks_12;
    struct ResBlock blocks_13;
    struct Upsample blocks_14;
    struct ResBlock blocks_15;
    struct ResBlock blocks_16;
    struct Upsample blocks_17;
    struct ResBlock blocks_18;
    struct ResBlock blocks_19;
    struct Upsample blocks_20;
    struct ResBlock blocks_21;
    struct ResBlock blocks_22;
    struct GroupNorm blocks_23;
    struct Conv2d blocks_24;

    void create_weight_tensors(struct ggml_context* ctx) {
        // (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        blocks_0.in_channels = 256;
        blocks_0.out_channels = 512;
        blocks_0.kernel_size = { 3, 3 };
        blocks_0.stride = { 1, 1 };
        blocks_0.padding = { 1, 1 };
        // blocks_0.has_bias = false;
        blocks_0.create_weight_tensors(ctx);

        // (1): ResBlock
        blocks_1.in_channels = 512;
        blocks_1.out_channels = 512;
        blocks_1.create_weight_tensors(ctx);

        // (2): AttnBlock
        blocks_2.in_channels = 512;
        blocks_2.create_weight_tensors(ctx);

        // (3-4): 2 x ResBlock
        blocks_3.in_channels = 512;
        blocks_3.out_channels = 512;
        blocks_3.create_weight_tensors(ctx);
        blocks_4.in_channels = 512;
        blocks_4.out_channels = 512;
        blocks_4.create_weight_tensors(ctx);

        // (5): AttnBlock
        blocks_5.in_channels = 512;
        blocks_5.create_weight_tensors(ctx);

        // (6): ResBlock
        blocks_6.in_channels = 512;
        blocks_6.out_channels = 512;
        blocks_6.create_weight_tensors(ctx);

        // (7): AttnBlock
        blocks_7.in_channels = 512;
        blocks_7.create_weight_tensors(ctx);

        // (8): Upsample
        blocks_8.in_channels = 512;
        blocks_8.create_weight_tensors(ctx);

        // (9): ResBlock
        blocks_9.in_channels = 512;
        blocks_9.out_channels = 256;
        blocks_9.create_weight_tensors(ctx);

        // (10): ResBlock
        blocks_10.in_channels = 256;
        blocks_10.out_channels = 256;
        blocks_10.create_weight_tensors(ctx);

        // (11): Upsample
        blocks_11.in_channels = 256;
        blocks_11.create_weight_tensors(ctx);

        // (12-13): 2 x ResBlock
        blocks_12.in_channels = 256;
        blocks_12.out_channels = 256;
        blocks_12.create_weight_tensors(ctx);
        blocks_13.in_channels = 256;
        blocks_13.out_channels = 256;
        blocks_13.create_weight_tensors(ctx);

        // (14): Upsample
        blocks_14.in_channels = 256;
        blocks_14.create_weight_tensors(ctx);

        // (15): ResBlock
        blocks_15.in_channels = 256;
        blocks_15.out_channels = 128;
        blocks_15.create_weight_tensors(ctx);

        // (16): ResBlock
        blocks_16.in_channels = 128;
        blocks_16.out_channels = 128;
        blocks_16.create_weight_tensors(ctx);

        // (17): Upsample
        blocks_17.in_channels = 128;
        blocks_17.create_weight_tensors(ctx);

        // (18-19): 2 x ResBlock
        blocks_18.in_channels = 128;
        blocks_18.out_channels = 128;
        blocks_18.create_weight_tensors(ctx);

        blocks_19.in_channels = 128;
        blocks_19.out_channels = 128;
        blocks_19.create_weight_tensors(ctx);


        // (20): Upsample
        blocks_20.in_channels = 128;
        blocks_20.create_weight_tensors(ctx);

        // (21): ResBlock
        blocks_21.in_channels = 128;
        blocks_21.out_channels = 64;
        blocks_21.create_weight_tensors(ctx);

        // (22): ResBlock
        blocks_22.in_channels = 64;
        blocks_22.out_channels = 64;
        blocks_22.create_weight_tensors(ctx);

        // (23): GroupNorm(32, 64, eps=1e-06, affine=True)
        blocks_23.num_channels = 64;
        blocks_23.eps = 1e-6;
        blocks_23.create_weight_tensors(ctx);

        // (24): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        blocks_24.in_channels = 64;
        blocks_24.out_channels = 3;
        blocks_24.kernel_size = { 3, 3 };
        blocks_24.stride = { 1, 1 };
        blocks_24.padding = { 1, 1 };
        // blocks_24.has_bias = false;
        blocks_24.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.0.");
        blocks_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.1.");
        blocks_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.2.");
        blocks_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.3.");
        blocks_3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.4.");
        blocks_4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.5.");
        blocks_5.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.6.");
        blocks_6.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.7.");
        blocks_7.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.8.");
        blocks_8.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.9.");
        blocks_9.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.10.");
        blocks_10.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.11.");
        blocks_11.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.12.");
        blocks_12.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.13.");
        blocks_13.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.14.");
        blocks_14.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.15.");
        blocks_15.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.16.");
        blocks_16.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.17.");
        blocks_17.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.18.");
        blocks_18.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.19.");
        blocks_19.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.20.");
        blocks_20.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.21.");
        blocks_21.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.22.");
        blocks_22.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.23.");
        blocks_23.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.24.");
        blocks_24.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
      x = blocks_0.forward(ctx, x);
      x = blocks_1.forward(ctx, x);
      x = blocks_2.forward(ctx, x);
      x = blocks_3.forward(ctx, x);
      x = blocks_4.forward(ctx, x);
      x = blocks_5.forward(ctx, x);
      x = blocks_6.forward(ctx, x);
      x = blocks_7.forward(ctx, x);
      x = blocks_8.forward(ctx, x);
      x = blocks_9.forward(ctx, x);
      x = blocks_10.forward(ctx, x);
      x = blocks_11.forward(ctx, x);
      x = blocks_12.forward(ctx, x);
      x = blocks_13.forward(ctx, x);
      x = blocks_14.forward(ctx, x);
      x = blocks_15.forward(ctx, x);
      x = blocks_16.forward(ctx, x);
      x = blocks_17.forward(ctx, x);
      x = blocks_18.forward(ctx, x);
      x = blocks_19.forward(ctx, x);
      x = blocks_20.forward(ctx, x);
      x = blocks_21.forward(ctx, x);
      x = blocks_22.forward(ctx, x);
      x = blocks_23.forward(ctx, x);
      x = blocks_24.forward(ctx, x);

    	return x;
    }
};


struct VectorQuantizer {
    ggml_tensor_t* embedding_weight;  // torch.float32, [1024, 256]

    void create_weight_tensors(struct ggml_context* ctx) {
        embedding_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1024);
    }

    void setup_weight_names(const char *prefix) {
        ggml_format_name(embedding_weight, "%s%s", prefix, "embedding.weight");
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!
      // xxxx_debug
    	return x;
    }
};


struct Downsample {
    int in_channels;
    int out_channels;

    // network params
    struct Conv2d conv;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

        conv.in_channels = in_channels;
        conv.out_channels = out_channels;
        conv.kernel_size = { 3, 3 };
        conv.stride = { 2, 2 };
        conv.padding = { 0, 0 };
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
      // pad = (0, 1, 0, 1)
      // x = F.pad(x, pad, mode="constant", value=0.0)
      // x = self.conv(x)
      // return x
      x = ggml_pad(ctx, x, 1, 1, 0, 0); // padding on H, W
      x = conv.forward(ctx, x);
    	return x;
    }
};


struct Encoder {
    struct Conv2d blocks_0;
    struct ResBlock blocks_1;
    struct ResBlock blocks_2;
    struct Downsample blocks_3;
    struct ResBlock blocks_4;
    struct ResBlock blocks_5;
    struct Downsample blocks_6;
    struct ResBlock blocks_7;
    struct ResBlock blocks_8;
    struct Downsample blocks_9;
    struct ResBlock blocks_10;
    struct ResBlock blocks_11;
    struct Downsample blocks_12;
    struct ResBlock blocks_13;
    struct ResBlock blocks_14;
    struct Downsample blocks_15;
    struct ResBlock blocks_16;
    struct AttnBlock blocks_17;
    struct ResBlock blocks_18;
    struct AttnBlock blocks_19;
    struct ResBlock blocks_20;
    struct AttnBlock blocks_21;
    struct ResBlock blocks_22;
    struct GroupNorm blocks_23;
    struct Conv2d blocks_24;

    void create_weight_tensors(struct ggml_context* ctx) {
        // (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        blocks_0.in_channels = 3;
        blocks_0.out_channels = 64;
        blocks_0.kernel_size = { 3, 3 };
        blocks_0.stride = { 1, 1 };
        blocks_0.padding = { 1, 1 };
        // blocks_0.has_bias = false;
        blocks_0.create_weight_tensors(ctx);

        // (1-2): 2 x ResBlock(
        blocks_1.in_channels = 64;
        blocks_1.out_channels = 64;
        blocks_1.create_weight_tensors(ctx);
        blocks_2.in_channels = 64;
        blocks_2.out_channels = 64;
        blocks_2.create_weight_tensors(ctx);

        // (3): Downsample
        blocks_3.in_channels = 64;
        blocks_3.create_weight_tensors(ctx);

        // (4): ResBlock
        blocks_4.in_channels = 64;
        blocks_4.out_channels = 128;
        blocks_4.create_weight_tensors(ctx);

        // (5): ResBlock
        blocks_5.in_channels = 128;
        blocks_5.out_channels = 128;
        blocks_5.create_weight_tensors(ctx);

        // (6): Downsample
        blocks_6.in_channels = 128;
        blocks_6.create_weight_tensors(ctx);

        // (7-8): 2 x ResBlock
        blocks_7.in_channels = 128;
        blocks_7.out_channels = 128;
        blocks_7.create_weight_tensors(ctx);
        blocks_8.in_channels = 128;
        blocks_8.out_channels = 128;
        blocks_8.create_weight_tensors(ctx);

        // (9): Downsample
        blocks_9.in_channels = 128;
        blocks_9.create_weight_tensors(ctx);

        // (10): ResBlock
        blocks_10.in_channels = 128;
        blocks_10.out_channels = 256;
        blocks_10.create_weight_tensors(ctx);

        // (11): ResBlock
        blocks_11.in_channels = 256;
        blocks_11.out_channels = 256;
        blocks_11.create_weight_tensors(ctx);

        // (12): Downsample
        blocks_12.in_channels = 256;
        blocks_12.create_weight_tensors(ctx);

        // (13-14): 2 x ResBlock
        blocks_13.in_channels = 256;
        blocks_13.out_channels = 256;
        blocks_13.create_weight_tensors(ctx);
        blocks_14.in_channels = 256;
        blocks_14.out_channels = 256;
        blocks_14.create_weight_tensors(ctx);

        // (15): Downsample
        blocks_15.in_channels = 256;
        blocks_15.create_weight_tensors(ctx);

        // (16): ResBlock
        blocks_16.in_channels = 256;
        blocks_16.out_channels = 512;
        blocks_16.create_weight_tensors(ctx);

        // (17): AttnBlock
        blocks_17.in_channels = 512;
        blocks_17.create_weight_tensors(ctx);

        // (18): ResBlock
        blocks_18.in_channels = 512;
        blocks_18.out_channels = 512;
        blocks_18.create_weight_tensors(ctx);

        // (19): AttnBlock
        blocks_19.in_channels = 512;
        blocks_19.create_weight_tensors(ctx);

        // (20): ResBlock
        blocks_20.in_channels = 512;
        blocks_20.out_channels = 512;
        blocks_20.create_weight_tensors(ctx);

        // (21): AttnBlock
        blocks_21.in_channels = 512;
        blocks_21.create_weight_tensors(ctx);

        // (22): ResBlock
        blocks_22.in_channels = 512;
        blocks_22.out_channels = 512;
        blocks_22.create_weight_tensors(ctx);

        // (23): GroupNorm(32, 512, eps=1e-06, affine=True)
        blocks_23.num_channels = 512;
        blocks_23.eps = 1e-6;
        blocks_23.create_weight_tensors(ctx);

        // (24): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        blocks_24.in_channels = 512;
        blocks_24.out_channels = 256;
        blocks_24.kernel_size = { 3, 3 };
        blocks_24.stride = { 1, 1 };
        blocks_24.padding = { 1, 1 };
        // blocks_24.has_bias = false;
        blocks_24.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.0.");
        blocks_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.1.");
        blocks_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.2.");
        blocks_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.3.");
        blocks_3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.4.");
        blocks_4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.5.");
        blocks_5.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.6.");
        blocks_6.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.7.");
        blocks_7.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.8.");
        blocks_8.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.9.");
        blocks_9.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.10.");
        blocks_10.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.11.");
        blocks_11.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.12.");
        blocks_12.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.13.");
        blocks_13.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.14.");
        blocks_14.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.15.");
        blocks_15.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.16.");
        blocks_16.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.17.");
        blocks_17.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.18.");
        blocks_18.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.19.");
        blocks_19.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.20.");
        blocks_20.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.21.");
        blocks_21.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.22.");
        blocks_22.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.23.");
        blocks_23.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "blocks.24.");
        blocks_24.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
      x = blocks_0.forward(ctx, x);
      x = blocks_1.forward(ctx, x);
      x = blocks_2.forward(ctx, x);
      x = blocks_3.forward(ctx, x);
      x = blocks_4.forward(ctx, x);
      x = blocks_5.forward(ctx, x);
      x = blocks_6.forward(ctx, x);
      x = blocks_7.forward(ctx, x);
      x = blocks_8.forward(ctx, x);
      x = blocks_9.forward(ctx, x);
      x = blocks_10.forward(ctx, x);
      x = blocks_11.forward(ctx, x);
      x = blocks_12.forward(ctx, x);
      x = blocks_13.forward(ctx, x);
      x = blocks_14.forward(ctx, x);
      x = blocks_15.forward(ctx, x);
      x = blocks_16.forward(ctx, x);
      x = blocks_17.forward(ctx, x);
      x = blocks_18.forward(ctx, x);
      x = blocks_19.forward(ctx, x);
      x = blocks_20.forward(ctx, x);
      x = blocks_21.forward(ctx, x);
      x = blocks_22.forward(ctx, x);
      x = blocks_23.forward(ctx, x);
      x = blocks_24.forward(ctx, x);

    	return x;
    }
};

struct CodeFormer {
    struct Encoder encoder;
    struct VectorQuantizer quantize;
    struct Generator generator;

    struct Linear feat_emb;

    struct TransformerSALayer ft_layers[9];

    struct LayerNorm idx_pred_layer_0;  // torch.float32, [512] 
    struct Linear idx_pred_layer_1;  // torch.float32, [1024, 512]

    void create_weight_tensors(struct ggml_context* ctx) {
        encoder.create_weight_tensors(ctx);
        quantize.create_weight_tensors(ctx);
        generator.create_weight_tensors(ctx);

        // self.feat_emb = nn.Linear(256, 512)
        feat_emb.in_features = 256;
        feat_emb.out_features = 512;
        feat_emb.create_weight_tensors(ctx);

        for (int i = 0; i < 9; i++) {
          ft_layers[i].create_weight_tensors(ctx);
        }

        // self.idx_pred_layer = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1024, bias=False))
        idx_pred_layer_0.normalized_shape = 512;
        idx_pred_layer_0.create_weight_tensors(ctx);

        idx_pred_layer_1.in_features = 512;
        idx_pred_layer_1.out_features = 1024;
        idx_pred_layer_1.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "encoder.");
        encoder.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "quantize.");
        quantize.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "generator.");
        generator.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "feat_emb.");
        feat_emb.setup_weight_names(s);

        for (int i = 0; i < 9; i++) {
            // snprintf(s, sizeof(s), "%s%s", prefix, "ft_layers.0.");
            snprintf(s, sizeof(s), "%sft_layers.%d.", prefix, i);
            ft_layers[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "idx_pred_layer.0");
        idx_pred_layer_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "idx_pred_layer.1");
        idx_pred_layer_1.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
    	// please implement forward by your self, please !!!
      // xxxx_debug
    	return x;
    }
};

#endif // __FACEGAN__H__
