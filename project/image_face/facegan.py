"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022-2024 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 13日 星期二 00:22:40 CST
# ***
# ************************************************************************************/
#
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

# from ggml_engine import create_network

import todos
import pdb

def load_facegan(model, path):
    """Load model."""
    cdir = os.path.dirname(__file__)
    path = path if cdir == "" else cdir + "/" + path

    if not os.path.exists(path):
        raise IOError(f"Model checkpoint '{path}' doesn't exist.")

    state_dict = torch.load(path, map_location=torch.device("cpu"))
    state_dict = state_dict["params_ema"]

    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n.startswith("fuse_convs_dict."):
            continue

        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)

def normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def swish(x):
    return x * torch.sigmoid(x)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert in_features == 512 and out_features == 512

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))


    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

def in_projection_packed(q, k, v, w, b) -> List[torch.Tensor]:
    # w.size() -- [1536, 512]
    # b.size() -- [1536]
    w_q, w_k, w_v = w.chunk(3)
    # (Pdb) w_q.size() -- [512, 512]
    # (Pdb) w_k.size() -- [512, 512]
    # (Pdb) w_v.size() -- [512, 512]

    b_q, b_k, b_v = b.chunk(3)
    # (Pdb) b_q.size(), b_k.size(), b_v.size() -- [512], [512], [512]

    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

def multi_head_attention_forward(query, key, value, num_heads: int,
    in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    head_dim = embed_dim // num_heads

    q, k, v = in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    #  pp q.size() -- [8, 256, 64]

    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    return attn_output


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        assert embed_dim == 512 and num_heads == 8

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # ==> 64
        self.in_proj_weight = nn.Parameter(torch.zeros((3 * embed_dim, embed_dim)))

        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = Linear(embed_dim, embed_dim)


    def forward(self, query, key, value):
        attn_output = multi_head_attention_forward(
            query, key, value, 
            self.num_heads,
            self.in_proj_weight, 
            self.in_proj_bias,
            self.out_proj.weight, 
            self.out_proj.bias,
        )
        return attn_output


class VectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1024, 256)

    def forward(self, indices, shape: List[int]):
        # input indices: batch*token_num -> (batch*token_num)*1
        b, c, n = indices.size() # (1, 256, 1)
        indices = indices.view(b * c * n, 1) # [256, 1]
        min_encodings = torch.zeros(b * c * n, 1024).to(indices)
        # torch.scatter_(input, dim, index, src)
        min_encodings.scatter_(1, indices, 1)
        # tensor [min_encodings] size: [256, 1024], min: 0.0, max: 1.0, mean: 0.000977

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.to(self.embedding.weight.dtype), self.embedding.weight)
        # z_q.size() -- [256, 256], self.embedding.weight.size() -- [1024, 256]
        # shape -- [1, 16, 16, 256]
        z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous() # [1, 16, 16, 256] -> [1, 256, 16, 16]
        # xxxx_debug
        return z_q


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_out = nn.Identity()

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)

        x_in = self.conv_out(x_in)

        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)

        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)
        w_ = w_.permute(0, 2, 1)

        # attend to values
        v = v.reshape(b, c, h * w)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(self, in_channels, nf=64):
        super().__init__()
        # in_channels = 3
        assert nf == 64

        ch_mult = [1, 2, 2, 4, 4, 8]
        self.num_resolutions = len(ch_mult)
        in_ch_mult = (1,) + tuple(ch_mult)
        # in_ch_mult == (1, 1, 2, 2, 4, 4, 8)

        blocks = []
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        assert self.num_resolutions == 6
        curr_res = 512
        for i in range(self.num_resolutions): # 6
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(2):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in [16]:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # non-local attention block
        assert block_in_ch == 512
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, 256, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class Generator(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        assert nf == 64

        ch_mult = [1, 2, 2, 4, 4, 8]
        self.num_resolutions = len(ch_mult) # 6 
        block_in_ch = nf * ch_mult[-1] # ==> 512
        curr_res = 16 # 512 // 2 ** (self.num_resolutions - 1) # 16

        blocks = []
        # initial conv
        blocks.append(nn.Conv2d(256, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)): # 6
            block_out_ch = nf * ch_mult[i]

            for _ in range(2): # 2
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in [16]:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, out_channels=3, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

def calc_mean_std(feat, eps: float = 1e-5) -> List[torch.Tensor]:
    """Calculate mean and std for adaptive_instance_normalization.
    """
    b, c, h, w = feat.size()
    feat_var = feat.view(b, c, h * w).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, h * w).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class TransformerSALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiheadAttention(512, 8)

        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(512, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

    def forward(self, target, query_pos: Optional[torch.Tensor]=None):
        # tensor [target] size: [256, 1, 512], min: -40.000423, max: 46.497059, mean: 0.031245
        # tensor [query_pos] size: [256, 1, 512], min: -0.896825, max: 0.903299, mean: 0.000741

        # self attention
        target2 = self.norm1(target)
        q = k = target2 + query_pos

        target2 = self.self_attn(q, k, value=target2)
        target = target + target2

        # ffn
        target2 = self.norm2(target)
        target2 = self.linear2(F.gelu(self.linear1(target2)))
        target = target + target2

        return target


class CodeFormer(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        assert nf == 64

        self.encoder = Encoder(3, nf)
        self.quantize = VectorQuantizer()
        self.generator = Generator(nf)

        fix_modules=["quantize", "generator"]
        for module in fix_modules:
            for param in getattr(self, module).parameters():
                param.requires_grad = False

        self.position_emb = nn.Parameter(torch.zeros(256, 512))
        self.feat_emb = nn.Linear(256, 512)

        # transformer
        self.ft_layers = nn.Sequential(
            *[TransformerSALayer()  for _ in range(9)] # n_layers -- 9
        )

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1024, bias=False))

        load_facegan(self, "models/codeformer.pth")
        torch.save(self.state_dict(), "/tmp/facegan.pth")

        # self.half()
        # pdb.set_trace()
        # create_network(self)

    def forward(self, x):
        # todos.debug.output_var("x1", x)
        # tensor [x1] size: [1, 3, 512, 512], min: 0.055676, max: 0.864457, mean: 0.235351

        x = (x - 0.5) / 0.5  # normal
        # ################### Encoder #####################
        x = self.encoder(x)
        # tensor [x] size: [1, 256, 16, 16], min: -2.690408, max: 2.767376, mean: -0.009793

        lq_feat = x
        # ################# Transformer ###################

        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        # [256, 512] --> [256, 1, 512]
        query_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1)) # BCHW -> BC(HW) -> (HW)BC
        # [1, 256, 16, 16] --> [1, 256, 256] --> [256, 1, 256] --> [256, 1, 512]

        # todos.debug.output_var("pos_emb", pos_emb)
        # todos.debug.output_var("query_emb 1", query_emb)
        # tensor [pos_emb] size: [256, 1, 512], min: -0.896825, max: 0.903299, mean: 0.000741
        # tensor [query_emb 1] size: [256, 1, 512], min: -40.000423, max: 46.497059, mean: 0.031245

        # Transformer encoder
        for layer in self.ft_layers: # 9 layers
            query_emb = layer(query_emb, query_pos=pos_emb)
        # tensor [query_emb] size: [256, 1, 512], min: -61.123913, max: 77.953194, mean: 0.027153

        # output logits
        logits = self.idx_pred_layer(query_emb)  # (HW)BC
        logits = logits.permute(1, 0, 2)  # (HW)BC -> B(HW)C
        # tensor [logits] size: [1, 256, 1024], min: -15.930029, max: 22.454041, mean: -1.539261

        soft_one_hot = F.softmax(logits, dim=2) # [1, 256, 1024]
        # GGML_API struct ggml_tensor * ggml_soft_max(
        #         struct ggml_context * ctx,
        #         struct ggml_tensor  * a);

        _, top_index = torch.topk(soft_one_hot, 1, dim=2)
        # // top k elements per row
        # GGML_API struct ggml_tensor * ggml_top_k(
        #         struct ggml_context * ctx,
        #         struct ggml_tensor  * a,
        #         int                   k);

        # todos.debug.output_var("top_index", top_index)
        # tensor [top_index] size: [1, 256, 1], min: 2.0, max: 1014.0, mean: 502.945312
        # top_index.dtype -- torch.int64
        quant_feat = self.quantize(top_index, shape=[x.shape[0], 16, 16, 256]) # shape [1, 16, 16, 256]
        # tensor [quant_feat] size: [1, 256, 16, 16], min: -2.466374, max: 2.514146, mean: -0.011886

        x = adaptive_instance_normalization(quant_feat, lq_feat)
        # tensor [x] size: [1, 256, 16, 16], min: -2.390994, max: 2.475082, mean: -0.009793

        # ################## Generator ####################
        out = self.generator(x)
        # todos.debug.output_var("out", out)
        # tensor [out] size: [1, 3, 512, 512], min: -0.956786, max: 0.922379, mean: -0.51932

        out = (out + 1.0) / 2.0  # change from [-1.0, 1.0] to [0.0, 1.0]

        return out.clamp(0.0, 1.0)
