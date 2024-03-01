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
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # in_features = 512
        # out_features = 512
        # bias = True

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
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        # embed_dim = 512
        # num_heads = 8

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.zeros((3 * embed_dim, embed_dim)))

        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)


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
    def __init__(self, codebook_size, emb_dim, beta):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        #### useless, place holder function ####
        return z

    def get_codebook_feat(self, indices, shape: List[int]):
        # input indices: batch*token_num -> (batch*token_num)*1

        b, c, n = indices.size()
        indices = indices.view(b * c * n, 1)
        min_encodings = torch.zeros(b * c * n, self.codebook_size).to(indices)

        min_encodings.scatter_(1, indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.to(self.embedding.weight.dtype), self.embedding.weight)
        z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

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
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
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
        self.in_channels = in_channels

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
    def __init__(self, in_channels, nf, emb_dim, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # initial convultion
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class Generator(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.ch_mult = ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)

        blocks = []
        # initial conv
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
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


class VQAutoEncoder(nn.Module):
    def __init__(self, 
        img_size=512, 
        nf=64, 
        ch_mult = [1, 2, 2, 4, 4, 8],
        res_blocks=2,
        attn_resolutions=[16],
        codebook_size=1024,
        emb_dim=256,
        beta=0.25,
    ):
        super().__init__()
        self.in_channels = 3
        self.nf = nf
        self.n_blocks = res_blocks
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions,
        )
        self.beta = beta  # 0.25
        self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
        self.generator = Generator(
            self.nf, self.embed_dim, self.ch_mult, self.n_blocks, self.resolution, self.attn_resolutions
        )

    def forward(self, x):
        #### useless, place holder function ####
        return x


def calc_mean_std(feat, eps: float = 1e-5) -> List[torch.Tensor]:
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, "The input feature should be 4D tensor."
    # b, c = size[:2]
    b, c, h, w = size
    feat_var = feat.view(b, c, h * w).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, h * w).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, nhead)

        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]): # make torch.jit.script happy
        # pos != None
        return tensor if pos is None else tensor + pos

    def forward(self, target, query_pos: Optional[torch.Tensor]=None):
        # self attention
        target2 = self.norm1(target)
        q = k = self.with_pos_embed(target2, query_pos)
        target2 = self.self_attn(q, k, value=target2)

        target = target + self.dropout1(target2)

        # ffn
        target2 = self.norm2(target)
        target2 = self.linear2(self.dropout(F.gelu(self.linear1(target2))))

        target = target + self.dropout2(target2)
        return target


class CodeFormer(VQAutoEncoder):
    def __init__(self,
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        fix_modules=["quantize", "generator"],
    ):
        super().__init__()

        for module in fix_modules:
            for param in getattr(self, module).parameters():
                param.requires_grad = False

        self.n_layers = n_layers
        self.position_emb = nn.Parameter(torch.zeros(latent_size, dim_embd))
        self.feat_emb = nn.Linear(256, dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(
            *[
                TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=dim_embd * 2, dropout=0.0)
                for _ in range(self.n_layers)
            ]
        )

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(dim_embd), nn.Linear(dim_embd, codebook_size, bias=False))

        load_facegan(self, "models/codeformer.pth")
        # self.half()

    # def on_cuda(self):
    #     return self.position_emb.is_cuda

    def forward(self, x):
        # if self.on_cuda():
        #     x = x.half()

        x = (x - 0.5) / 0.5  # normal
        # ################### Encoder #####################
        x = self.encoder(x)

        lq_feat = x
        # ################# Transformer ###################
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        query_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1)) # BCHW -> BC(HW) -> (HW)BC

        # Transformer encoder
        for layer in self.ft_layers: # 9 layers
            query_emb = layer(query_emb, query_pos=pos_emb)

        # output logits
        logits = self.idx_pred_layer(query_emb)  # (hw)bn
        logits = logits.permute(1, 0, 2)  # (hw)bn -> b(hw)n

        soft_one_hot = F.softmax(logits, dim=2)
        _, top_index = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_index, shape=[x.shape[0], 16, 16, 256])

        x = adaptive_instance_normalization(quant_feat, lq_feat)

        # ################## Generator ####################
        out = self.generator(x)

        out = (out + 1.0) / 2.0  # change from [-1.0, 1.0] to [0.0, 1.0]

        return out.clamp(0.0, 1.0).float()
