#ifndef __FACEGAN__H__
#define __FACEGAN__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>

#pragma GCC diagnostic ignored "-Wformat-truncation"

// ggml_set_name(out, "xxxx_test");
// ggml_set_output(out);
struct MultiheadAttention {
    int num_heads = 8;
    int embed_dim = 512;

    struct Linear in_proj;
    struct Linear out_proj;

    void create_weight_tensors(ggml_context_t* ctx) {
        in_proj.in_features = embed_dim;
        in_proj.out_features = 3 * embed_dim;
        in_proj.create_weight_tensors(ctx, GGML_TYPE_F32);

        out_proj.in_features = embed_dim;
        out_proj.out_features = embed_dim;
        out_proj.create_weight_tensors(ctx, GGML_TYPE_F32);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "in_proj_");
        in_proj.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "out_proj.");
        out_proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, ggml_tensor_t* query, ggml_tensor_t* key, ggml_tensor_t* value) {
        // query    f32 [512, 1, 256, 1], 
        // key    f32 [512, 1, 256, 1], 
        // value    f32 [512, 1, 256, 1], 
        // in_proj.weight    f32 [512, 1536, 1, 1] 
        // in_proj.bias    f32 [1536, 1, 1, 1]

        int QB = (int)query->ne[2]; // query batch -- 256
        int KB = (int)key->ne[2]; // key batch -- 256
        int VB = (int)value->ne[2]; // value batch -- 256
        int head_dim = embed_dim / num_heads; // 64

        std::vector<ggml_tensor_t *> w_qkv = ggml_nn_chunks(ctx, in_proj.weight, 1, 3);
        std::vector<ggml_tensor_t *> b_qkv = ggml_nn_chunks(ctx, in_proj.bias, 0, 3);
        ggml_tensor_t *w_q = w_qkv[0];
        ggml_tensor_t *w_k = w_qkv[1];
        ggml_tensor_t *w_v = w_qkv[2];
        // ---------------------------------------------------
        ggml_tensor_t *b_q = b_qkv[0];
        ggml_tensor_t *b_k = b_qkv[1];
        ggml_tensor_t *b_v = b_qkv[2];

        ggml_tensor_t *g_q = ggml_nn_linear(ctx, query, w_q, b_q);
        g_q = ggml_cont(ctx, ggml_reshape_4d(ctx, g_q, head_dim /*32*/, num_heads /*8*/, QB /*100*/, 1)); // head_dim -- 32
        g_q = ggml_cont(ctx, ggml_permute(ctx, g_q, 0, 2, 1, 3)); // [64, 8, 256, 1] -> [64, 256, 8, 1]

        ggml_tensor_t *g_q_scaled = ggml_scale(ctx, g_q, 1.0/sqrtf(head_dim /*32*/)); // head_dim -- 32

        ggml_tensor_t *g_k = ggml_nn_linear(ctx, key, w_k, b_k); // f32 [512, 1, 256, 1]
        g_k = ggml_cont(ctx, ggml_reshape_4d(ctx, g_k, head_dim /*64*/, num_heads /*8*/, KB /*256*/, 1));
        g_k = ggml_cont(ctx, ggml_permute(ctx, g_k, 0, 2, 1, 3));
        // g_k: [512, 1, 256, 1] --> [64, 8, 256, 1] --> [256, 64, 8, 1]

        ggml_tensor_t *g_v = ggml_nn_linear(ctx, value, w_v, b_v); // f32 [512, 1, 256, 1]
        g_v = ggml_cont(ctx, ggml_reshape_4d(ctx, g_v, head_dim /*64*/, num_heads /*8*/, VB /*256*/, 1));
        g_v = ggml_cont(ctx, ggml_permute(ctx, g_v, 0, 2, 1, 3));
        // g_v: [512, 1, 256, 1] --> [64, 8, 256, 1] --> [64, 256, 8, 1]

        ggml_tensor_t *attn_output_weights = 
            ggml_nn_mul_mat(ctx, g_q_scaled, ggml_transpose(ctx, g_k));
        // [64, 256, 8, 1] x [64, 256, 8, 1] --> [256, 256, 8, 1]
        attn_output_weights = ggml_soft_max(ctx, attn_output_weights);
        // ------------------------------------------------------------------------

        ggml_tensor_t *attn_output = ggml_nn_mul_mat(ctx, attn_output_weights, g_v);
        // tensor [attn_output] size: [8, 256, 64], min: -1.775282, max: 2.079419, mean: 0.007756
        attn_output = ggml_cont(ctx, ggml_permute(ctx, attn_output, 0, 2, 1, 3));
        attn_output = ggml_cont(ctx, ggml_reshape_2d(ctx, attn_output, embed_dim /*512*/, QB /*256*/));
        // [64, 256, 8, 1] -> [64, 8, 256, 1] --> [512, 256]

        attn_output = ggml_nn_linear(ctx, attn_output, out_proj.weight, out_proj.bias);
        // tensor [attn_output] size: [256, 512], min: -7.639149, max: 10.361456, mean: 0.002153

        attn_output = ggml_cont(ctx, ggml_reshape_3d(ctx, attn_output, embed_dim /*512*/, 1, QB /*256*/));
        // tensor [attn_output4] size: [256, 1, 512], min: -7.639149, max: 10.361456, mean: 0.002153

        return attn_output; // f32 [256, 1, 512, 1]
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
    // network params
    struct MultiheadAttention self_attn; // xxxx_debug

    struct Linear linear1;
    struct Linear linear2;

    struct LayerNorm norm1;
    struct LayerNorm norm2;

    void create_weight_tensors(struct ggml_context* ctx) {
        self_attn.create_weight_tensors(ctx);

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

        snprintf(s, sizeof(s), "%s%s", prefix, "self_attn.");
        self_attn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "linear1.");
        linear1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "linear2.");
        linear2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x, ggml_tensor_t* pos) {
        // target2 = self.norm1(target)
        // q = k = target2 + query_pos
        ggml_tensor_t* x2 = norm1.forward(ctx, x);
        ggml_tensor_t* q = ggml_add(ctx, x2, pos);
        ggml_tensor_t* k = q;

        // target2 = self.self_attn(q, k, value=target2)
        // target = target + target2
        x2 = self_attn.forward(ctx, q, k, x2);
        x = ggml_add(ctx, x, x2);

        // ffn
        // target2 = self.norm2(target)
        // target2 = self.linear2(F.gelu(self.linear1(target2)))
        // target = target + target2

        x2 = norm2.forward(ctx, x);
        x2 = linear1.forward(ctx, x2);
        x2 = ggml_relu(ctx, x2);
        x2 = linear2.forward(ctx, x2);

        x = ggml_add(ctx, x, x2);

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
        conv.create_weight_tensors(ctx);
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

// --------------------------------------------------
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
        norm1.eps = 1e-6;
        norm1.create_weight_tensors(ctx);

        conv1.in_channels = in_channels;
        conv1.out_channels = out_channels;
        conv1.kernel_size = { 3, 3 };
        conv1.stride = { 1, 1 };
        conv1.padding = { 1, 1 };
        conv1.create_weight_tensors(ctx);

        norm2.num_channels = out_channels;
        norm2.eps = 1e-6;
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

// ----------------------------------------------------------
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
        // # tensor [x] size: [1, 512, 16, 16], min: -255.195648, max: 226.06636, mean: -0.000203
        ggml_tensor_t *h_x = x;
        h_x = norm.forward(ctx, h_x);
        // # tensor [h_x] size: [1, 512, 16, 16], min: -3.551194, max: 5.92871, mean: -0.010494

        ggml_tensor_t* q_x = q.forward(ctx, h_x);
        ggml_tensor_t* k_x = k.forward(ctx, h_x);
        ggml_tensor_t* v_x = v.forward(ctx, h_x);
        // # tensor [q_x] size: [1, 512, 16, 16], min: -8.109866, max: 5.720445, mean: -0.021126
        // # tensor [k_x] size: [1, 512, 16, 16], min: -8.596012, max: 7.313824, mean: -0.015812
        // # tensor [v_x] size: [1, 512, 16, 16], min: -7.816355, max: 9.079727, mean: -0.007039

        int B = (int)q_x->ne[3];
        int C = (int)q_x->ne[2]; // === 512
        int H = (int)q_x->ne[1]; // === 16
        int W = (int)q_x->ne[0]; // === 16

        q_x = ggml_reshape_3d(ctx, q_x, H*W, C, B);
        q_x = ggml_permute(ctx, q_x, 1, 0, 2, 3); // [HW, C, B, 1] --> [C, HW, B, 1]
        q_x = ggml_cont(ctx, q_x);

        k_x = ggml_reshape_3d(ctx, k_x, H*W, C, B);
        // # tensor [k_x] size: [1, 512, 256], min: -8.596012, max: 7.313824, mean: -0.015812

        ggml_tensor_t *w_x = ggml_nn_mul_mat(ctx, q_x, k_x);
        w_x = ggml_scale(ctx, w_x, 1.0/sqrtf(C));
        w_x = ggml_soft_max(ctx, w_x);
        // tensor [w_x] size: [1, 256, 256], min: 0.002891, max: 0.005134, mean: 0.003906
        w_x = ggml_permute(ctx, w_x, 1, 0, 2, 3); // [HW1, HW2, B, 1] --> [HW2, HW1, B, 1]

        v_x = ggml_reshape_3d(ctx, v_x, H*W, C, B);
        h_x = ggml_nn_mul_mat(ctx, v_x, w_x);
        h_x = ggml_reshape_4d(ctx, h_x, W, H, C, B);
        h_x = proj_out.forward(ctx, h_x);

        x = ggml_add(ctx, h_x, x);

        return x; //  f32 [16, 16, 512, 1]
    }
};

struct Generator {
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

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* index) {
        index = ggml_cont(ctx, index);
        index = ggml_reshape_1d(ctx, index, 256);
        ggml_tensor_t *min_encodings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 256);
        // min_encodings = ggml_clamp(ctx, min_encodings, 0.0, 0.0);
        min_encodings = ggml_constant(ctx, min_encodings, 0.0);
        min_encodings = ggml_scatter(ctx, min_encodings, 0 /*dim --> 1024 */, index);
        min_encodings = ggml_cont(ctx, min_encodings);
        ggml_tensor_t *z_q = ggml_nn_mul_mat(ctx, min_encodings, embedding_weight);

        z_q = ggml_reshape_4d(ctx, z_q, 256, 16, 16, 1);
        z_q = ggml_permute(ctx, z_q, 2, 0, 1, 3); // [256, 16, 16, 1] --> [16, 16, 256, 1]
        z_q = ggml_cont(ctx, z_q);

        return z_q;
    }
};

// -----------------------------------------
struct Downsample {
    int in_channels;
    // int out_channels;

    struct Conv2d conv;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = in_channels;
        conv.out_channels = in_channels;
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
        x = ggml_pad(ctx, x, 1, 1, 0, 0); // padding on H, W
        x = conv.forward(ctx, x);
        // x = ggml_cont(ctx, x); // !!!

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

        // (1-2): 2 x ResBlock
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

// def calc_mean_std(feat, eps: float = 1e-5) -> List[torch.Tensor]:
//     """Calculate mean and std for instance_normal.
//     """
//     b, c, h, w = feat.size()
//     feat_var = feat.view(b, c, h * w).var(dim=2) + eps
//     feat_std = feat_var.sqrt().view(b, c, 1, 1)

//     feat_mean = feat.view(b, c, h * w).mean(dim=2).view(b, c, 1, 1)
//     return feat_mean, feat_std


// def instance_normal(content_feat, style_feat):
//     """Adaptive instance normalization.
//     """
//     size = content_feat.size()
//     style_mean, style_std = calc_mean_std(style_feat)
//     content_mean, content_std = calc_mean_std(content_feat)
//     normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
//     return normalized_feat * style_std.expand(size) + style_mean.expand(size)

static ggml_tensor_t *feat_mean(struct ggml_context* ctx, ggml_tensor_t *f)
{
    int B = (int)f->ne[3];
    int C = (int)f->ne[2];
    int H = (int)f->ne[1];
    int W = (int)f->ne[0];
    f = ggml_reshape_3d(ctx, f, H*W, C, B);
    ggml_tensor_t *m = ggml_nn_mean(ctx, f, 0/*dim on HW*/);

    return ggml_reshape_4d(ctx, m, 1, 1, C, B);
}

static ggml_tensor_t *feat_std(struct ggml_context* ctx, ggml_tensor_t *f)
{
    int B = (int)f->ne[3];
    int C = (int)f->ne[2];
    int H = (int)f->ne[1];
    int W = (int)f->ne[0];
    f = ggml_reshape_3d(ctx, f, H*W, C, B);
    ggml_tensor_t *s = ggml_nn_std(ctx, f, 0/*dim on HW*/, 1e-5);

    return ggml_reshape_4d(ctx, s, 1, 1, C, B);
}


static ggml_tensor_t *instance_normal(struct ggml_context* ctx, ggml_tensor_t *c_feat, ggml_tensor_t *s_feat)
{
//     style_mean, style_std = calc_mean_std(style_feat)
//     content_mean, content_std = calc_mean_std(content_feat)
//     normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
//     return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    ggml_tensor_t *c_mean = feat_mean(ctx, c_feat);
    ggml_tensor_t *c_std = feat_std(ctx, c_feat);

    ggml_tensor_t *s_mean = feat_mean(ctx, s_feat);
    ggml_tensor_t *s_std = feat_std(ctx, s_feat);

    ggml_tensor_t *n = ggml_sub(ctx, c_feat, c_mean);
    n = ggml_div(ctx, n, c_std);

    n = ggml_mul(ctx, n, s_std);
    n = ggml_add(ctx, n, s_mean);

    return n;
}

struct CodeFormer : GGMLNetwork {
    struct Encoder encoder;
    struct VectorQuantizer quantize;
    struct Generator generator;

    ggml_tensor_t *position_emb;
    struct Linear feat_emb;

    struct TransformerSALayer ft_layers[9];

    struct LayerNorm idx_pred_layer_0;  // torch.float32, [512] 
    struct Linear idx_pred_layer_1;  // torch.float32, [1024, 512]

    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 8; // 2048 * 4
    }

    void create_weight_tensors(struct ggml_context* ctx) {
        encoder.create_weight_tensors(ctx);
        quantize.create_weight_tensors(ctx);
        generator.create_weight_tensors(ctx);

        position_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 256);

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
        idx_pred_layer_1.has_bias = false;
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

        ggml_format_name(position_emb, "%s%s", prefix, "position_emb");

        snprintf(s, sizeof(s), "%s%s", prefix, "feat_emb.");
        feat_emb.setup_weight_names(s);

        for (int i = 0; i < 9; i++) {
            // snprintf(s, sizeof(s), "%s%s", prefix, "ft_layers.0.");
            snprintf(s, sizeof(s), "%sft_layers.%d.", prefix, i);
            ft_layers[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "idx_pred_layer.0.");
        idx_pred_layer_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "idx_pred_layer.1.");
        idx_pred_layer_1.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_UNUSED(argc);

        ggml_tensor_t* x = argv[0];
        // tensor [x] size: [1, 3, 512, 512], min: 0.055676, max: 0.864457, mean: 0.235351

        // x = (x - 0.5)/0.5
        // x = ggml_nn_add(ctx, x, -0.5);
        x = ggml_add_constant(ctx, x, -0.5);
        x = ggml_scale(ctx, x, 2.0); 

        x = encoder.forward(ctx, x);
        // tensor [x] size: [1, 256, 16, 16], min: -2.690408, max: 2.767376, mean: -0.009793

        ggml_tensor_t *pos_emb = ggml_reshape_3d(ctx, position_emb, 512, 1, 256);
        // tensor [pos_emb] size: [256, 1, 512], min: -0.896825, max: 0.903299, mean: 0.000741

        ggml_tensor_t *lq_feat = x; // tensor [x] size: [1, 256, 16, 16]
        ggml_tensor_t *t = ggml_reshape_4d(ctx, lq_feat, 256 /*HW*/, 256 /*C*/, 1/*B*/, 1); // BCHW --> BC(HW)
        t = ggml_permute(ctx, t, 2, 0, 1, 3); // BC(HW) --> (HW)BC
        ggml_tensor_t *query_emb = feat_emb.forward(ctx, ggml_cont(ctx, t));
        // tensor [query_emb] size: [256, 1, 512], min: -40.000423, max: 46.497059, mean: 0.031245

        for (int i = 0; i < 9; i++) {
            query_emb = ft_layers[i].forward(ctx, query_emb, pos_emb);
        }
        // tensor [query_emb] size: [256, 1, 512], min: -53.13633, max: 61.400383, mean: 0.052259

        // logits = self.idx_pred_layer(query_emb)  # (HW)BC
        ggml_tensor_t *logits = idx_pred_layer_0.forward(ctx, query_emb);
        logits = idx_pred_layer_1.forward(ctx, logits);
        logits = ggml_cont(ctx, ggml_permute(ctx, logits, 0, 2, 1, 3));
        // # tensor [logits] size: [1, 256, 1024], min: -15.930029, max: 22.454041, mean: -1.539261

        ggml_tensor_t *soft_one_hot = ggml_soft_max(ctx, logits); // [1024, 256, 1]
        ggml_tensor_t *top_index = ggml_top_k(ctx, soft_one_hot, 1 /*k*/);
        // # tensor [top_index] size: [1, 256, 1], min: 2.0, max: 1014.0, mean: 502.945312

        ggml_tensor_t *quant_feat = quantize.forward(ctx, top_index);
        // # tensor [quant_feat] size: [1, 256, 16, 16], min: -2.466374, max: 2.514146, mean: -0.011886

        x = instance_normal(ctx, quant_feat, lq_feat);
        // # tensor [x] size: [1, 256, 16, 16], min: -2.390994, max: 2.475082, mean: -0.009793

        // # ################## Generator ####################
        ggml_tensor_t *out = generator.forward(ctx, x);
        // # tensor [out] size: [1, 3, 512, 512], min: -0.956786, max: 0.922379, mean: -0.51932

        // out = (out + 1.0) / 2.0  # change from [-1.0, 1.0] to [0.0, 1.0]
        // out = ggml_nn_add(ctx, out, 1.0); // 
        out = ggml_add_constant(ctx, out, 1.0);
        out = ggml_scale(ctx, out, 0.5);
        out = ggml_clamp(ctx, out, 0.0, 1.0);

        return out;
    }
};

#endif // __FACEGAN__H__
