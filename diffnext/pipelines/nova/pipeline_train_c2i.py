# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""NOVA C2I training pipeline."""

from typing import Dict

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch

from diffnext.engine import engine_utils
from diffnext.pipelines.builder import build_diffusion_scheduler
from diffnext.pipelines.pipeline_utils import PipelineMixin


class NOVATrainC2IPipeline(DiffusionPipeline, PipelineMixin):
    """Pipeline for training NOVA C2I models."""

    _optional_components = ["transformer", "scheduler", "vae"]

    def __init__(self, transformer=None, scheduler=None, vae=None, trust_remote_code=True):
        super(NOVATrainC2IPipeline, self).__init__()
        self.vae = self.register_module(vae, "vae")
        self.transformer = self.register_module(transformer, "transformer")
        self.scheduler = self.register_module(scheduler, "scheduler")
        self.transformer.noise_scheduler = build_diffusion_scheduler(self.scheduler)
        self.transformer.sample_scheduler, self.guidance_scale = self.scheduler, 5.0

    @property
    def model(self) -> torch.nn.Module:
        """Return the trainable model."""
        return self.transformer

    def configure_model(self, config) -> torch.nn.Module:
        """Configure the trainable model."""
        ckpt_lvl = config.model.get("gradient_checkpointing", 0)
        self.model.loss_repeat = config.model.get("loss_repeat", 4)
        [setattr(blk, "mlp_checkpointing", ckpt_lvl) for blk in self.model.video_encoder.blocks]
        for blk in self.model.image_encoder.blocks if hasattr(self.model, "image_encoder") else []:
            setattr(blk, "mlp_checkpointing", ckpt_lvl > 1)
        [setattr(blk, "mlp_checkpointing", ckpt_lvl > 2) for blk in self.model.image_decoder.blocks]
        engine_utils.freeze_module(self.model.label_embed.norm)  # We always use frozen LN.
        engine_utils.freeze_module(self.model.video_pos_embed)  # Freeze it during C2I.
        engine_utils.freeze_module(self.model.video_encoder.patch_embed)  # Freeze it during C2I.
        self.model.pipeline_preprocess = self.preprocess
        return self.model.train()

    def prepare_latents(self, inputs: Dict):
        """Prepare the video latents."""
        if "images" in inputs:
            raise NotImplementedError
        elif "latents" in inputs:
            x = torch.as_tensor(inputs.pop("latents"), device=self.device)
            x = x.to(dtype=self.dtype if x.is_floating_point() else torch.int64)
            inputs["x"] = self.vae.scale_(self.vae.latent_dist(x).sample())

    def encode_prompt(self, inputs: Dict):
        """Encode class prompts."""
        prompts = torch.as_tensor(inputs.pop("prompt"), device=self.device)
        inputs["c"] = [self.transformer.label_embed(prompts)]

    def preprocess(self, inputs: Dict) -> Dict:
        """Define the pipeline preprocess at every call."""
        if not self.model.training:
            raise RuntimeError("Excepted a trainable model.")
        self.prepare_latents(inputs)
        self.encode_prompt(inputs)
