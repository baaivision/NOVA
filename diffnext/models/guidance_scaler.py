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
"""Classifier-free guidance scaler."""

import torch


class GuidanceScaler(object):
    """Guidance scaler."""

    def __init__(self, **kwargs):
        self.guidance_scale = kwargs.get("guidance_scale", 1)
        self.guidance_trunc = kwargs.get("guidance_trunc", 0)
        self.guidance_renorm = kwargs.get("guidance_renorm", 1)
        self.image_guidance_scale = kwargs.get("image_guidance_scale", 0)
        self.spatiotemporal_guidance_scale = kwargs.get("spatiotemporal_guidance_scale", 0)
        self.min_guidance_scale = kwargs.get("min_guidance_scale", None) or self.guidance_scale
        self.inc_guidance_scale = self.guidance_scale - self.min_guidance_scale

    @property
    def extra_pass(self) -> bool:
        """Return if an additional (third) guidance pass is required."""
        return self.image_guidance_scale + self.spatiotemporal_guidance_scale > 0

    def clone(self):
        """Return a deepcopy of current guidance scaler."""
        return GuidanceScaler(**self.__dict__)

    def decay_guidance_scale(self, decay=0):
        """Scale guidance scale according to decay."""
        self.guidance_scale = self.inc_guidance_scale * decay + self.min_guidance_scale

    def expand(self, x: torch.Tensor, padding: torch.Tensor = None) -> torch.Tensor:
        """Expand input tensor for guidance passes."""
        x = torch.stack([x] * (3 if self.extra_pass else 2)) if self.guidance_scale > 1 else x
        x.__setitem__(1, padding) if self.image_guidance_scale and padding is not None else None
        return x.flatten(0, 1) if self.guidance_scale > 1 else x

    def expand_text(self, c: torch.Tensor) -> torch.Tensor:
        """Expand text embedding tensor for guidance passes."""
        c = list(c.chunk(2)) if self.extra_pass else c
        c.append(c[1]) if self.image_guidance_scale else None  # Null, Null
        c.append(c[0]) if self.spatiotemporal_guidance_scale else None  # Null, Text
        return torch.cat(c) if self.extra_pass else c

    def maybe_disable(self, timestep, *args):
        """Disable all guidance passes if matching truncation threshold."""
        if self.guidance_scale > 1 and self.guidance_trunc:
            if float(timestep) < self.guidance_trunc:
                self.guidance_scale = 1
                return [_.chunk(3 if self.extra_pass else 2)[0] for _ in args]
        return args

    def renorm(self, x, cond):
        """Apply guidance renormalization to input logits."""
        if self.guidance_renorm >= 1:
            return x
        args = {"dim": tuple(range(1, len(x.shape))), "keepdim": True}
        return x.mul_(cond.norm(**args).div_(x.norm(**args)).clamp(self.guidance_renorm, 1))

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Apply guidance passes to input logits."""
        if self.guidance_scale <= 1:
            return x
        if self.image_guidance_scale:
            cond, uncond, imgcond = x.chunk(3)
            x = self.renorm(uncond.add(cond.sub(imgcond).mul_(self.guidance_scale)), cond)
            return x.add_(imgcond.sub_(uncond).mul_(self.image_guidance_scale))
        if self.spatiotemporal_guidance_scale:
            cond, uncond, perturb = x.chunk(3)
            x = self.renorm(uncond.add_(cond.sub(uncond).mul_(self.guidance_scale)), cond)
            return x.add_(cond.sub_(perturb).mul_(self.spatiotemporal_guidance_scale))
        cond, uncond = x.chunk(2)
        return self.renorm(uncond.add_(cond.sub(uncond).mul_(self.guidance_scale)), cond)
