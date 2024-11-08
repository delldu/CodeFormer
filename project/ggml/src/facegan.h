#ifndef __FACEGAN__H__
#define __FACEGAN__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

/*
 Linear(in_features=512, out_features=512, bias=True) */

struct Linear {
    // network hparams
    int in_features = 512;
    int out_features = 512;

    // network params
    


    void create_weight_tensors(struct ggml_context* ctx) {
        
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

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

    void setup_weight_names(char *prefix) {
        char s[512];
        snprintf(s, sizeof(s), "%s%s", prefix, "out_proj.");
        out_proj.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

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
    struct MultiheadAttention self_attn;
    struct ggml_tensor* linear1_weight;  // torch.float32, [1024, 512] 
    struct ggml_tensor* linear1_bias;  // torch.float32, [1024] 
    struct ggml_tensor* linear2_weight;  // torch.float32, [512, 1024] 
    struct ggml_tensor* linear2_bias;  // torch.float32, [512] 
    struct ggml_tensor* norm1_weight;  // torch.float32, [512] 
    struct ggml_tensor* norm1_bias;  // torch.float32, [512] 
    struct ggml_tensor* norm2_weight;  // torch.float32, [512] 
    struct ggml_tensor* norm2_bias;  // torch.float32, [512]


    void create_weight_tensors(struct ggml_context* ctx) {
        self_attn.create_weight_tensors(ctx);
        linear1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1024);
        linear1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
        linear2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 512);
        linear2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        norm1_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        norm1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        norm2_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        norm2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        snprintf(s, sizeof(s), "%s%s", prefix, "self_attn.");
        self_attn.setup_weight_names(s);
        ggml_format_name(linear1_weight, "%s%s", prefix, "linear1.weight");
        ggml_format_name(linear1_bias, "%s%s", prefix, "linear1.bias");
        ggml_format_name(linear2_weight, "%s%s", prefix, "linear2.weight");
        ggml_format_name(linear2_bias, "%s%s", prefix, "linear2.bias");
        ggml_format_name(norm1_weight, "%s%s", prefix, "norm1.weight");
        ggml_format_name(norm1_bias, "%s%s", prefix, "norm1.bias");
        ggml_format_name(norm2_weight, "%s%s", prefix, "norm2.weight");
        ggml_format_name(norm2_bias, "%s%s", prefix, "norm2.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 Upsample(
  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
) */

struct Upsample {
    // network hparams
    

    // network params
    struct ggml_tensor* conv_weight;  // torch.float32, [128, 128, 3, 3] 
    struct ggml_tensor* conv_bias;  // torch.float32, [128]


    void create_weight_tensors(struct ggml_context* ctx) {
        conv_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 128, 128);
        conv_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        ggml_format_name(conv_weight, "%s%s", prefix, "conv.weight");
        ggml_format_name(conv_bias, "%s%s", prefix, "conv.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 Generator(
  (blocks): ModuleList(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ResBlock(
      (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (2): AttnBlock(
      (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
      (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
    )
    (3-4): 2 x ResBlock(
      (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (5): AttnBlock(
      (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
      (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
    )
    (6): ResBlock(
      (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (7): AttnBlock(
      (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
      (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
    )
    (8): Upsample(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (9): ResBlock(
      (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    )
    (10): ResBlock(
      (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (11): Upsample(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (12-13): 2 x ResBlock(
      (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (14): Upsample(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (15): ResBlock(
      (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    )
    (16): ResBlock(
      (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (17): Upsample(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (18-19): 2 x ResBlock(
      (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (20): Upsample(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (21): ResBlock(
      (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 64, eps=1e-06, affine=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
    )
    (22): ResBlock(
      (norm1): GroupNorm(32, 64, eps=1e-06, affine=True)
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 64, eps=1e-06, affine=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (23): GroupNorm(32, 64, eps=1e-06, affine=True)
    (24): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
) */

struct Generator {
    // network hparams
    int num_resolutions = 6;

    // network params
    struct ggml_tensor* blocks_0_weight;  // torch.float32, [512, 256, 3, 3] 
    struct ggml_tensor* blocks_0_bias;  // torch.float32, [512] 
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
    struct ggml_tensor* blocks_23_weight;  // torch.float32, [64] 
    struct ggml_tensor* blocks_23_bias;  // torch.float32, [64] 
    struct ggml_tensor* blocks_24_weight;  // torch.float32, [3, 64, 3, 3] 
    struct ggml_tensor* blocks_24_bias;  // torch.float32, [3]


    void create_weight_tensors(struct ggml_context* ctx) {
        blocks_0_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 256, 512);
        blocks_0_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        blocks_1.create_weight_tensors(ctx);
        blocks_2.create_weight_tensors(ctx);
        blocks_3.create_weight_tensors(ctx);
        blocks_4.create_weight_tensors(ctx);
        blocks_5.create_weight_tensors(ctx);
        blocks_6.create_weight_tensors(ctx);
        blocks_7.create_weight_tensors(ctx);
        blocks_8.create_weight_tensors(ctx);
        blocks_9.create_weight_tensors(ctx);
        blocks_10.create_weight_tensors(ctx);
        blocks_11.create_weight_tensors(ctx);
        blocks_12.create_weight_tensors(ctx);
        blocks_13.create_weight_tensors(ctx);
        blocks_14.create_weight_tensors(ctx);
        blocks_15.create_weight_tensors(ctx);
        blocks_16.create_weight_tensors(ctx);
        blocks_17.create_weight_tensors(ctx);
        blocks_18.create_weight_tensors(ctx);
        blocks_19.create_weight_tensors(ctx);
        blocks_20.create_weight_tensors(ctx);
        blocks_21.create_weight_tensors(ctx);
        blocks_22.create_weight_tensors(ctx);
        blocks_23_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        blocks_23_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        blocks_24_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 64, 3);
        blocks_24_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        ggml_format_name(blocks_0_weight, "%s%s", prefix, "blocks.0.weight");
        ggml_format_name(blocks_0_bias, "%s%s", prefix, "blocks.0.bias");
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
        ggml_format_name(blocks_23_weight, "%s%s", prefix, "blocks.23.weight");
        ggml_format_name(blocks_23_bias, "%s%s", prefix, "blocks.23.bias");
        ggml_format_name(blocks_24_weight, "%s%s", prefix, "blocks.24.weight");
        ggml_format_name(blocks_24_bias, "%s%s", prefix, "blocks.24.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 VectorQuantizer(
  (embedding): Embedding(1024, 256)
) */

struct VectorQuantizer {
    // network hparams
    

    // network params
    struct ggml_tensor* embedding_weight;  // torch.float32, [1024, 256]


    void create_weight_tensors(struct ggml_context* ctx) {
        embedding_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1024);
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        ggml_format_name(embedding_weight, "%s%s", prefix, "embedding.weight");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

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
    // network hparams
    

    // network params
    struct ggml_tensor* norm_weight;  // torch.float32, [512] 
    struct ggml_tensor* norm_bias;  // torch.float32, [512] 
    struct ggml_tensor* q_weight;  // torch.float32, [512, 512, 1, 1] 
    struct ggml_tensor* q_bias;  // torch.float32, [512] 
    struct ggml_tensor* k_weight;  // torch.float32, [512, 512, 1, 1] 
    struct ggml_tensor* k_bias;  // torch.float32, [512] 
    struct ggml_tensor* v_weight;  // torch.float32, [512, 512, 1, 1] 
    struct ggml_tensor* v_bias;  // torch.float32, [512] 
    struct ggml_tensor* proj_out_weight;  // torch.float32, [512, 512, 1, 1] 
    struct ggml_tensor* proj_out_bias;  // torch.float32, [512]


    void create_weight_tensors(struct ggml_context* ctx) {
        norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        q_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 512, 512);
        q_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        k_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 512, 512);
        k_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        v_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 512, 512);
        v_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        proj_out_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 512, 512);
        proj_out_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        ggml_format_name(norm_weight, "%s%s", prefix, "norm.weight");
        ggml_format_name(norm_bias, "%s%s", prefix, "norm.bias");
        ggml_format_name(q_weight, "%s%s", prefix, "q.weight");
        ggml_format_name(q_bias, "%s%s", prefix, "q.bias");
        ggml_format_name(k_weight, "%s%s", prefix, "k.weight");
        ggml_format_name(k_bias, "%s%s", prefix, "k.bias");
        ggml_format_name(v_weight, "%s%s", prefix, "v.weight");
        ggml_format_name(v_bias, "%s%s", prefix, "v.bias");
        ggml_format_name(proj_out_weight, "%s%s", prefix, "proj_out.weight");
        ggml_format_name(proj_out_bias, "%s%s", prefix, "proj_out.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 Downsample(
  (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
) */

struct Downsample {
    // network hparams
    

    // network params
    struct ggml_tensor* conv_weight;  // torch.float32, [256, 256, 3, 3] 
    struct ggml_tensor* conv_bias;  // torch.float32, [256]


    void create_weight_tensors(struct ggml_context* ctx) {
        conv_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 256, 256);
        conv_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        ggml_format_name(conv_weight, "%s%s", prefix, "conv.weight");
        ggml_format_name(conv_bias, "%s%s", prefix, "conv.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 ResBlock(
  (norm1): GroupNorm(32, 64, eps=1e-06, affine=True)
  (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (norm2): GroupNorm(32, 64, eps=1e-06, affine=True)
  (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv_out): Identity()
) */

struct ResBlock {
    // network hparams
    

    // network params
    struct ggml_tensor* norm1_weight;  // torch.float32, [64] 
    struct ggml_tensor* norm1_bias;  // torch.float32, [64] 
    struct ggml_tensor* conv1_weight;  // torch.float32, [64, 64, 3, 3] 
    struct ggml_tensor* conv1_bias;  // torch.float32, [64] 
    struct ggml_tensor* norm2_weight;  // torch.float32, [64] 
    struct ggml_tensor* norm2_bias;  // torch.float32, [64] 
    struct ggml_tensor* conv2_weight;  // torch.float32, [64, 64, 3, 3] 
    struct ggml_tensor* conv2_bias;  // torch.float32, [64]


    void create_weight_tensors(struct ggml_context* ctx) {
        norm1_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        norm1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        conv1_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 64, 64);
        conv1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        norm2_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        norm2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        conv2_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 64, 64);
        conv2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        ggml_format_name(norm1_weight, "%s%s", prefix, "norm1.weight");
        ggml_format_name(norm1_bias, "%s%s", prefix, "norm1.bias");
        ggml_format_name(conv1_weight, "%s%s", prefix, "conv1.weight");
        ggml_format_name(conv1_bias, "%s%s", prefix, "conv1.bias");
        ggml_format_name(norm2_weight, "%s%s", prefix, "norm2.weight");
        ggml_format_name(norm2_bias, "%s%s", prefix, "norm2.bias");
        ggml_format_name(conv2_weight, "%s%s", prefix, "conv2.weight");
        ggml_format_name(conv2_bias, "%s%s", prefix, "conv2.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 Encoder(
  (blocks): ModuleList(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1-2): 2 x ResBlock(
      (norm1): GroupNorm(32, 64, eps=1e-06, affine=True)
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 64, eps=1e-06, affine=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (3): Downsample(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2))
    )
    (4): ResBlock(
      (norm1): GroupNorm(32, 64, eps=1e-06, affine=True)
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
    )
    (5): ResBlock(
      (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (6): Downsample(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
    )
    (7-8): 2 x ResBlock(
      (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (9): Downsample(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
    )
    (10): ResBlock(
      (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
    )
    (11): ResBlock(
      (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (12): Downsample(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
    )
    (13-14): 2 x ResBlock(
      (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (15): Downsample(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
    )
    (16): ResBlock(
      (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
    )
    (17): AttnBlock(
      (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
      (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
    )
    (18): ResBlock(
      (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (19): AttnBlock(
      (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
      (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
    )
    (20): ResBlock(
      (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (21): AttnBlock(
      (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
      (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
    )
    (22): ResBlock(
      (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv_out): Identity()
    )
    (23): GroupNorm(32, 512, eps=1e-06, affine=True)
    (24): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
) */

struct Encoder {
    // network hparams
    int num_resolutions = 6;

    // network params
    struct ggml_tensor* blocks_0_weight;  // torch.float32, [64, 3, 3, 3] 
    struct ggml_tensor* blocks_0_bias;  // torch.float32, [64] 
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
    struct ggml_tensor* blocks_23_weight;  // torch.float32, [512] 
    struct ggml_tensor* blocks_23_bias;  // torch.float32, [512] 
    struct ggml_tensor* blocks_24_weight;  // torch.float32, [256, 512, 3, 3] 
    struct ggml_tensor* blocks_24_bias;  // torch.float32, [256]


    void create_weight_tensors(struct ggml_context* ctx) {
        blocks_0_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 3, 64);
        blocks_0_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        blocks_1.create_weight_tensors(ctx);
        blocks_2.create_weight_tensors(ctx);
        blocks_3.create_weight_tensors(ctx);
        blocks_4.create_weight_tensors(ctx);
        blocks_5.create_weight_tensors(ctx);
        blocks_6.create_weight_tensors(ctx);
        blocks_7.create_weight_tensors(ctx);
        blocks_8.create_weight_tensors(ctx);
        blocks_9.create_weight_tensors(ctx);
        blocks_10.create_weight_tensors(ctx);
        blocks_11.create_weight_tensors(ctx);
        blocks_12.create_weight_tensors(ctx);
        blocks_13.create_weight_tensors(ctx);
        blocks_14.create_weight_tensors(ctx);
        blocks_15.create_weight_tensors(ctx);
        blocks_16.create_weight_tensors(ctx);
        blocks_17.create_weight_tensors(ctx);
        blocks_18.create_weight_tensors(ctx);
        blocks_19.create_weight_tensors(ctx);
        blocks_20.create_weight_tensors(ctx);
        blocks_21.create_weight_tensors(ctx);
        blocks_22.create_weight_tensors(ctx);
        blocks_23_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        blocks_23_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        blocks_24_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 512, 256);
        blocks_24_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        ggml_format_name(blocks_0_weight, "%s%s", prefix, "blocks.0.weight");
        ggml_format_name(blocks_0_bias, "%s%s", prefix, "blocks.0.bias");
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
        ggml_format_name(blocks_23_weight, "%s%s", prefix, "blocks.23.weight");
        ggml_format_name(blocks_23_bias, "%s%s", prefix, "blocks.23.bias");
        ggml_format_name(blocks_24_weight, "%s%s", prefix, "blocks.24.weight");
        ggml_format_name(blocks_24_bias, "%s%s", prefix, "blocks.24.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

/*
 CodeFormer(
  (encoder): Encoder(
    (blocks): ModuleList(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1-2): 2 x ResBlock(
        (norm1): GroupNorm(32, 64, eps=1e-06, affine=True)
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 64, eps=1e-06, affine=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (3): Downsample(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2))
      )
      (4): ResBlock(
        (norm1): GroupNorm(32, 64, eps=1e-06, affine=True)
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
      )
      (5): ResBlock(
        (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (6): Downsample(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
      )
      (7-8): 2 x ResBlock(
        (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (9): Downsample(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
      )
      (10): ResBlock(
        (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (11): ResBlock(
        (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (12): Downsample(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
      )
      (13-14): 2 x ResBlock(
        (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (15): Downsample(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
      )
      (16): ResBlock(
        (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (17): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (18): ResBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (19): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (20): ResBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (21): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (22): ResBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (23): GroupNorm(32, 512, eps=1e-06, affine=True)
      (24): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  (quantize): VectorQuantizer(
    (embedding): Embedding(1024, 256)
  )
  (generator): Generator(
    (blocks): ModuleList(
      (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ResBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (2): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (3-4): 2 x ResBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (5): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (6): ResBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (7): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (8): Upsample(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (9): ResBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (10): ResBlock(
        (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (11): Upsample(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (12-13): 2 x ResBlock(
        (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (14): Upsample(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (15): ResBlock(
        (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
        (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      )
      (16): ResBlock(
        (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (17): Upsample(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (18-19): 2 x ResBlock(
        (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (20): Upsample(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (21): ResBlock(
        (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
        (conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 64, eps=1e-06, affine=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (22): ResBlock(
        (norm1): GroupNorm(32, 64, eps=1e-06, affine=True)
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 64, eps=1e-06, affine=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv_out): Identity()
      )
      (23): GroupNorm(32, 64, eps=1e-06, affine=True)
      (24): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  (feat_emb): Linear(in_features=256, out_features=512, bias=True)
  (ft_layers): Sequential(
    (0): TransformerSALayer(
      (self_attn): MultiheadAttention(
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (linear1): Linear(in_features=512, out_features=1024, bias=True)
      (linear2): Linear(in_features=1024, out_features=512, bias=True)
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (1): TransformerSALayer(
      (self_attn): MultiheadAttention(
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (linear1): Linear(in_features=512, out_features=1024, bias=True)
      (linear2): Linear(in_features=1024, out_features=512, bias=True)
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (2): TransformerSALayer(
      (self_attn): MultiheadAttention(
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (linear1): Linear(in_features=512, out_features=1024, bias=True)
      (linear2): Linear(in_features=1024, out_features=512, bias=True)
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (3): TransformerSALayer(
      (self_attn): MultiheadAttention(
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (linear1): Linear(in_features=512, out_features=1024, bias=True)
      (linear2): Linear(in_features=1024, out_features=512, bias=True)
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (4): TransformerSALayer(
      (self_attn): MultiheadAttention(
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (linear1): Linear(in_features=512, out_features=1024, bias=True)
      (linear2): Linear(in_features=1024, out_features=512, bias=True)
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (5): TransformerSALayer(
      (self_attn): MultiheadAttention(
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (linear1): Linear(in_features=512, out_features=1024, bias=True)
      (linear2): Linear(in_features=1024, out_features=512, bias=True)
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (6): TransformerSALayer(
      (self_attn): MultiheadAttention(
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (linear1): Linear(in_features=512, out_features=1024, bias=True)
      (linear2): Linear(in_features=1024, out_features=512, bias=True)
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (7): TransformerSALayer(
      (self_attn): MultiheadAttention(
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (linear1): Linear(in_features=512, out_features=1024, bias=True)
      (linear2): Linear(in_features=1024, out_features=512, bias=True)
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (8): TransformerSALayer(
      (self_attn): MultiheadAttention(
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (linear1): Linear(in_features=512, out_features=1024, bias=True)
      (linear2): Linear(in_features=1024, out_features=512, bias=True)
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
  )
  (idx_pred_layer): Sequential(
    (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (1): Linear(in_features=512, out_features=1024, bias=False)
  )
) */

struct CodeFormer {
    // network hparams
    

    // network params
    struct Encoder encoder;
    struct VectorQuantizer quantize;
    struct Generator generator;
    struct ggml_tensor* feat_emb_weight;  // torch.float32, [512, 256] 
    struct ggml_tensor* feat_emb_bias;  // torch.float32, [512] 
    struct TransformerSALayer ft_layers_0;
    struct TransformerSALayer ft_layers_1;
    struct TransformerSALayer ft_layers_2;
    struct TransformerSALayer ft_layers_3;
    struct TransformerSALayer ft_layers_4;
    struct TransformerSALayer ft_layers_5;
    struct TransformerSALayer ft_layers_6;
    struct TransformerSALayer ft_layers_7;
    struct TransformerSALayer ft_layers_8;
    struct ggml_tensor* idx_pred_layer_0_weight;  // torch.float32, [512] 
    struct ggml_tensor* idx_pred_layer_0_bias;  // torch.float32, [512] 
    struct ggml_tensor* idx_pred_layer_1_weight;  // torch.float32, [1024, 512]


    void create_weight_tensors(struct ggml_context* ctx) {
        encoder.create_weight_tensors(ctx);
        quantize.create_weight_tensors(ctx);
        generator.create_weight_tensors(ctx);
        feat_emb_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 512);
        feat_emb_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        ft_layers_0.create_weight_tensors(ctx);
        ft_layers_1.create_weight_tensors(ctx);
        ft_layers_2.create_weight_tensors(ctx);
        ft_layers_3.create_weight_tensors(ctx);
        ft_layers_4.create_weight_tensors(ctx);
        ft_layers_5.create_weight_tensors(ctx);
        ft_layers_6.create_weight_tensors(ctx);
        ft_layers_7.create_weight_tensors(ctx);
        ft_layers_8.create_weight_tensors(ctx);
        idx_pred_layer_0_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        idx_pred_layer_0_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        idx_pred_layer_1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1024);
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        snprintf(s, sizeof(s), "%s%s", prefix, "encoder.");
        encoder.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "quantize.");
        quantize.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "generator.");
        generator.setup_weight_names(s);
        ggml_format_name(feat_emb_weight, "%s%s", prefix, "feat_emb.weight");
        ggml_format_name(feat_emb_bias, "%s%s", prefix, "feat_emb.bias");
        snprintf(s, sizeof(s), "%s%s", prefix, "ft_layers.0.");
        ft_layers_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ft_layers.1.");
        ft_layers_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ft_layers.2.");
        ft_layers_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ft_layers.3.");
        ft_layers_3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ft_layers.4.");
        ft_layers_4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ft_layers.5.");
        ft_layers_5.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ft_layers.6.");
        ft_layers_6.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ft_layers.7.");
        ft_layers_7.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ft_layers.8.");
        ft_layers_8.setup_weight_names(s);
        ggml_format_name(idx_pred_layer_0_weight, "%s%s", prefix, "idx_pred_layer.0.weight");
        ggml_format_name(idx_pred_layer_0_bias, "%s%s", prefix, "idx_pred_layer.0.bias");
        ggml_format_name(idx_pred_layer_1_weight, "%s%s", prefix, "idx_pred_layer.1.weight");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    	// please implement forward by your self, please !!!

    	return x;
    }
};

#endif // __FACEGAN__H__
