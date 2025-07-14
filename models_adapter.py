# modified from: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py

from typing import Tuple
from collections import OrderedDict
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import configs
from configs import (
    CLIP_VIT_B16_PATH,
    CLIP_VIT_L14_PATH,
    DWCONV3D_DISABLE_CUDNN,
    )

gpu_id = configs.gpu_id
device = torch.device("cuda:" + str(gpu_id))

class Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size1,kernel_size2,kernel_size3):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        #in=768，out=384

        self.conv1 = nn.Conv3d(adapter_channels, adapter_channels,kernel_size=kernel_size1,stride=(1, 1, 1),padding=tuple(x // 2 for x in kernel_size1),groups=adapter_channels)
        self.conv2 = nn.Conv3d(adapter_channels, adapter_channels,kernel_size=kernel_size2,stride=(1, 1, 1),padding=tuple(x // 2 for x in kernel_size2),groups=adapter_channels)
        self.conv3 = nn.Conv3d(adapter_channels, adapter_channels,kernel_size=kernel_size3,stride=(1, 1, 1),padding=tuple(x // 2 for x in kernel_size3),groups=adapter_channels)
        self.projector = nn.Conv3d(adapter_channels, adapter_channels,kernel_size=kernel_size1,stride=(1, 1, 1),padding=tuple(x // 2 for x in kernel_size1),groups=adapter_channels)
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.conv1.weight, 0.)
        nn.init.constant_(self.conv1.bias, 0.)
        nn.init.constant_(self.conv2.weight, 0.)
        nn.init.constant_(self.conv2.bias, 0.)
        nn.init.constant_(self.conv3.weight, 0.)
        nn.init.constant_(self.conv3.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, T):
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv1.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x#128.197.768
        x = x[:, 1:, :]#128.196.768
        x = self.fc1(x)#16.384.768
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()#16.8.14.14.384--》16.384.8.14.14

        cudnn_enabled = torch.backends.cudnn.enabled#True
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        identity = x
        conv1_x = self.conv1(x)#16.384.8.14.14
        conv2_x = self.conv2(x)#16.384.8.14.14
        conv3_x = self.conv3(x)#16.384.8.14.14
        x = (conv1_x + conv2_x + conv3_x) / 3.0
        x=x+ identity
        identity = x

        x = self.projector(x)
        x=identity + x
        torch.backends.cudnn.enabled = cudnn_enabled
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)#128.196.384
        x = self.fc2(x)#128.196.768
        x_id[:, 1:, :] += x

        return x_id


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_bias=attn_bias.to(device)
    # print("attn_bias",attn_bias.is_cuda)

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 adapter_width: int,
                 adapter_kernel_size1: Tuple[int, int, int],
                 adapter_kernel_size2: Tuple[int, int, int],
                 adapter_kernel_size3: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        adapter_class = functools.partial(
            Adapter,
            in_channels=d_model,
            adapter_channels=adapter_width,
            kernel_size1=adapter_kernel_size1,
            kernel_size2=adapter_kernel_size2,
            kernel_size3=adapter_kernel_size3,
        )
        self.adapter_pre_attn = \
            adapter_class() if adapter_pre_attn else None
        self.adapter_pre_mlp = \
            adapter_class() if adapter_pre_mlp else None

    # Efficient implementation equivalent to the following:

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()
        H = self.attn.num_heads

        qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([H, H, H], dim=1)
        # out = F.scaled_dot_product_attention(q, k, v)
        out = scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).flatten(-2)
        out = self.attn.out_proj(out)

        return out
        

    def forward(self,
                x: torch.Tensor,
                num_frames: int
                ) -> torch.Tensor:
        if self.adapter_pre_attn is not None:
            x = self.adapter_pre_attn(x, num_frames)
        x = x + self.attention(self.ln_1(x))
        if self.adapter_pre_mlp is not None:
            x = self.adapter_pre_mlp(x, num_frames)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size1: Tuple[int, int, int],
                 adapter_kernel_size2: Tuple[int, int, int],
                 adapter_kernel_size3: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                adapter_width=adapter_width,
                adapter_kernel_size1=adapter_kernel_size1,
                adapter_kernel_size2=adapter_kernel_size2,
                adapter_kernel_size3=adapter_kernel_size3,
                adapter_pre_attn=adapter_pre_attn and i >= layers - adapter_layers,
                adapter_pre_mlp=adapter_pre_mlp and i >= layers - adapter_layers,
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, num_frames)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 num_classes: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size1: Tuple[int, int, int],
                 adapter_kernel_size2: Tuple[int, int, int],
                 adapter_kernel_size3: Tuple[int, int, int],
                 adapter_pre_attn: bool,
                 adapter_pre_mlp: bool,
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
            kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                (input_resolution // patch_size) ** 2 + 1, width
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads,
            adapter_width, adapter_layers, adapter_kernel_size1,adapter_kernel_size2,adapter_kernel_size3,
            adapter_pre_attn, adapter_pre_mlp)

        self.ln_post = LayerNorm(width)

        for n, p in self.named_parameters():
          if 'adapter' not in n:
            p.requires_grad_(False)
            p.data = p.data.half()
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(width, num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0.)


    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        spatial_size = tuple(x.size()[2:])
        x = x.flatten(-2).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x
            ], dim=1)  # [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.view(B, T, x.size(1), x.size(2)).flatten(0, 1) # BT, L, D

        x = self.transformer(x, T)

        x = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))
        x = x[:, :, 0, :].mean(dim=1)

        x = self.ln_post(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def clip_vit_base_patch16_adapter24x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size1=(3, 1, 1),
        adapter_kernel_size2=(5, 1, 1),
        adapter_kernel_size3=(7, 1, 1),
        adapter_pre_attn=True,
        adapter_pre_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py.'
    # checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='device')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model


def clip_vit_base_patch16_adapter12x384(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_layers=12,
        adapter_kernel_size1=(3, 3, 3),
        adapter_kernel_size2=(5, 5, 5),
        adapter_kernel_size3=(7, 7, 7),
        # adapter_pre_attn=True,
        adapter_pre_attn=False,
        adapter_pre_mlp=True,
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    # checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cuda')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    print("已引入参数，这里")
    return model
