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
"""Exponential Moving Average (EMA) of model updates."""

import copy
import torch


class ModelEMA(torch.nn.Module):
    """Model Exponential Moving Average."""

    def __init__(self, model, decay=0.99, update_every=100, device="gpu"):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.model = copy.deepcopy(model).eval()
        self.model._apply(lambda t: t.float() if t.requires_grad else t)  # FP32.
        [setattr(p, "requires_grad", False) for p in self.model.parameters()]
        self.model.cpu() if device == "cpu" else None

    @torch.no_grad()
    def update(self, model):
        for ema_v, model_v in zip(self.model.parameters(), model.parameters()):
            if not model_v.requires_grad:
                continue
            new_value = model_v.data.float()
            value = ema_v.to(device=new_value.device)
            ema_v.copy_(value.mul_(self.decay).add_(new_value, alpha=1 - self.decay))
