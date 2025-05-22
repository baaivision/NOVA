# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Diffusion Transformer."""

from functools import partial
from typing import Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as apply_ckpt

from diffnext.models.embeddings import PatchEmbed, RotaryEmbed3D
from diffnext.models.normalization import AdaLayerNormZero, AdaLayerNormSingle
from diffnext.models.diffusion_mlp import Projector, TimeCondEmbed


class TimeEmbed(TimeCondEmbed):
    """Time embedding layer."""

    def __init__(self, embed_dim, freq_dim=256):
        nn.Module.__init__(self)
        self.timestep_proj = Projector(freq_dim, embed_dim, embed_dim)
        self.freq_dim, self.time_freq = freq_dim, None

    def forward(self, timestep) -> torch.Tensor:
        dtype = self.timestep_proj.fc1.weight.dtype
        temb = self.timestep_proj(self.get_freq_embed(timestep, dtype))
        return temb.unsqueeze_(1) if temb.dim() == 2 else temb


class MLP(nn.Module):
    """Two layers MLP."""

    def __init__(self, dim, mlp_ratio=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.activation = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class Attention(nn.Module):
    """Multihead attention."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super(Attention, self).__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj, self.pe_func = nn.Linear(dim, dim), None

    def forward(self, x) -> torch.Tensor:
        qkv_shape = [-1, x.size(1), 3, self.num_heads, self.head_dim]
        q, k, v = self.qkv(x).view(qkv_shape).permute(2, 0, 3, 1, 4).unbind(dim=0)
        q, k = (self.pe_func(q), self.pe_func(k)) if self.pe_func else (q, k)
        o = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).flatten(2))


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=True, modulation_type=None):
        super(Block, self).__init__()
        self.modulation = (modulation_type or AdaLayerNormZero)(dim, num_stats=6, eps=1e-6)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)
        self.attn_checkpointing = self.mlp_checkpointing = self.stg_skip = False

    def forward_modulation(self, x, z) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        return self.modulation(x, z)

    def forward_attn(self, x) -> torch.Tensor:
        return self.norm1(self.attn(x))

    def forward_mlp(self, x) -> torch.Tensor:
        return self.norm2(self.mlp(x))

    def forward_ckpt(self, x, name) -> torch.Tensor:
        if getattr(self, f"{name}_checkpointing", False) and x.requires_grad:
            return apply_ckpt(getattr(self, f"forward_{name}"), x, use_reentrant=False)
        return getattr(self, f"forward_{name}")(x)

    def forward(self, x, z, pe_func: callable = None) -> torch.Tensor:
        self.attn.pe_func = pe_func
        stg_x = x.chunk(3)[-1] if self.stg_skip else None
        if self.mlp_checkpointing and x.requires_grad:
            x, stats = apply_ckpt(self.forward_modulation, x, z, use_reentrant=False)
        else:
            x, stats = self.forward_modulation(x, z)
        gate_msa, scale_mlp, shift_mlp, gate_mlp = stats
        x = self.forward_ckpt(x, "attn").mul(gate_msa).add_(x)
        x = self.modulation.norm(x).mul(1 + scale_mlp).add_(shift_mlp)
        x = self.forward_ckpt(x, "mlp").mul(gate_mlp).add_(x)
        return torch.cat(x.chunk(3)[:2] + (stg_x,)) if self.stg_skip else x


class DiffusionTransformer(nn.Module):
    """Diffusion transformer."""

    def __init__(
        self,
        depth,
        embed_dim,
        num_heads,
        mlp_ratio=4,
        patch_size=2,
        image_size=32,
        image_dim=None,
        modulation=True,
    ):
        super(DiffusionTransformer, self).__init__()
        final_norm = AdaLayerNormSingle if modulation else AdaLayerNormZero
        block = partial(Block, modulation_type=AdaLayerNormSingle) if modulation else Block
        self.embed_dim, self.image_size, self.image_dim = embed_dim, image_size, image_dim
        self.patch_embed = PatchEmbed(image_dim, embed_dim, patch_size)
        self.time_embed = TimeEmbed(embed_dim)
        self.modulation = AdaLayerNormZero(embed_dim, num_stats=6, eps=1e-6) if modulation else None
        self.rope = RotaryEmbed3D(embed_dim // num_heads)
        self.blocks = nn.ModuleList(block(embed_dim, num_heads, mlp_ratio) for _ in range(depth))
        self.norm = final_norm(embed_dim, num_stats=2, eps=1e-6)
        self.head = nn.Linear(embed_dim, patch_size**2 * image_dim)

    def prepare_pe(self, c=None, pos=None) -> Tuple[callable, callable]:
        return self.rope.get_func(pos, pad=0 if c is None else c.size(1))

    def forward(self, x, timestep, c=None, pos=None) -> torch.Tensor:
        x = self.patch_embed(x)
        t = self.time_embed(timestep)
        z = self.modulation.proj(self.modulation.activation(t)) if self.modulation else t
        pe = self.prepare_pe(c, pos) if pos is not None else None
        x = x if c is None else torch.cat([c, x], dim=1)
        for blk in self.blocks:
            x = blk(x, z, pe)
        x = self.norm(x if c is None else x[:, c.size(1) :], t)[0]
        return self.head(x)
