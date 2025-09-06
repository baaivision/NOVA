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
"""Flex attention layers."""

from itertools import accumulate
from typing import List

import torch
from torch import nn

try:
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    flex_attention = create_block_mask = None


class FlexAttentionCausal2D(nn.Module):
    """Block-wise causal flex attention."""

    def __init__(self):
        super(FlexAttentionCausal2D, self).__init__()
        self.attn_func = self.offsets = self.flags = None
        self.cu_offsets = self.block_mask = None

    def set_offsets(self, offsets: List[int]):
        """Set block-wise mask offsets."""
        offsets = list(type(offsets)([0]) + offsets if offsets[0] != 0 else offsets)
        if offsets != self.offsets:
            self.offsets, self.block_mask = offsets, None

    def set_offsets_by_lens(self, lens, flags=None):
        """Set block-wise mask offsets by lengths."""
        self.set_offsets(list(accumulate(type(lens)([0]) + lens if lens[0] != 0 else lens)))
        self.flags = flags  # Bidirectional flags.

    def get_mask_mod(self) -> callable:
        """Return the mask modification."""
        counts = self.cu_offsets[1:] - self.cu_offsets[:-1]
        ids = torch.arange(len(counts), device=self.cu_offsets.device, dtype=torch.int32)
        ids = ids.repeat_interleave(counts)
        if self.flags is None:
            return lambda b, h, qi, ki: (qi >= ki) | (ids[qi] == ids[ki])
        flags = list(self.flags) + [0] * (len(counts) - len(self.flags))
        flags = torch.as_tensor(flags, device=self.cu_offsets.device, dtype=torch.bool)
        flags = flags.repeat_interleave(counts)
        return lambda b, h, qi, ki: (qi >= ki) | (ids[qi] == ids[ki] and flags[qi])

    def get_attn_func(self) -> callable:
        """Return the attention function."""
        if flex_attention is None:
            raise NotImplementedError(f"FlexAttn requires torch>=2.5 but got {torch.__version__}")
        if self.attn_func is None:
            self.attn_func = torch.compile(flex_attention)
        return self.attn_func

    def get_block_mask(self, q: torch.Tensor) -> torch.Tensor:
        """Return the attention block mask according to inputs."""
        if self.block_mask is not None:
            return self.block_mask
        b, h, q_len = q.shape[:3]
        args = {"B": b, "H": h, "Q_LEN": q_len, "KV_LEN": q_len, "_compile": True}
        self.cu_offsets = torch.as_tensor(self.offsets, device=q.device, dtype=torch.int32)
        self.block_mask = create_block_mask(self.get_mask_mod(), **args)
        return self.block_mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.get_attn_func()(q, k, v, block_mask=self.get_block_mask(q), enable_gqa=True)
