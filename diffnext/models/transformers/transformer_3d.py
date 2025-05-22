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
"""Base 3D transformer model for video generation."""

from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

from diffnext.models.guidance_scaler import GuidanceScaler


class Transformer3DModel(nn.Module):
    """Base 3D transformer model for video generation."""

    def __init__(
        self,
        video_encoder=None,
        image_encoder=None,
        image_decoder=None,
        mask_embed=None,
        text_embed=None,
        label_embed=None,
        video_pos_embed=None,
        image_pos_embed=None,
        motion_embed=None,
        noise_scheduler=None,
        sample_scheduler=None,
    ):
        super(Transformer3DModel, self).__init__()
        self.video_encoder = video_encoder
        self.image_encoder = image_encoder
        self.image_decoder = image_decoder
        self.mask_embed = mask_embed
        self.text_embed = text_embed
        self.label_embed = label_embed
        self.video_pos_embed = video_pos_embed
        self.image_pos_embed = image_pos_embed
        self.motion_embed = motion_embed
        self.noise_scheduler = noise_scheduler
        self.sample_scheduler = sample_scheduler
        self.pipeline_preprocess = lambda inputs: inputs
        self.loss_repeat = 4

    def progress_bar(self, iterable, enable=True):
        """Return a tqdm progress bar."""
        return tqdm(iterable) if enable else iterable

    def preprocess(self, inputs: Dict):
        """Preprocess model inputs."""
        add_guidance = inputs.get("guidance_scale", 1) > 1
        inputs["c"], dtype, device = inputs.get("c", []), self.dtype, self.device
        if inputs.get("x", None) is None:
            batch_size = inputs.get("batch_size", 1)
            image_size = (self.image_encoder.image_dim,) + self.image_encoder.image_size
            inputs["x"] = torch.empty(batch_size, *image_size, device=device, dtype=dtype)
        if inputs.get("prompt", None) is not None and self.text_embed:
            inputs["c"].append(self.text_embed(inputs.pop("prompt")))
        if inputs.get("motion_flow", None) is not None and self.motion_embed:
            flow, fps = inputs.pop("motion_flow", None), inputs.pop("fps", None)
            flow, fps = [v + v if (add_guidance and v) else v for v in (flow, fps)]
            inputs["c"].append(self.motion_embed(inputs["c"][-1], flow, fps))
        inputs["c"] = torch.cat(inputs["c"], dim=1) if len(inputs["c"]) > 1 else inputs["c"][0]

    def get_losses(self, z: torch.Tensor, x: torch.Tensor, video_shape=None) -> Dict:
        """Return the training losses."""
        z = z.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        x = x.repeat(self.loss_repeat, *((1,) * (x.dim() - 1)))
        x = self.image_encoder.patch_embed.patchify(x)
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        timestep = self.noise_scheduler.sample_timesteps(z.shape[:2], device=z.device)
        x_t = self.noise_scheduler.add_noise(x, noise, timestep)
        x_t = self.image_encoder.patch_embed.unpatchify(x_t)
        timestep = getattr(self.noise_scheduler, "timestep", timestep)
        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "flow")
        model_pred = self.image_decoder(x_t, timestep, z)
        model_target = noise.float() if pred_type == "epsilon" else noise.sub(x).float()
        loss = nn.functional.mse_loss(model_pred.float(), model_target, reduction="none")
        loss, weight = loss.mean(-1, True), self.mask_embed.mask.to(loss.dtype)
        weight = weight.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        loss = loss.mul_(weight).div_(weight.sum().add_(1e-5))
        if video_shape is not None:
            loss = loss.view((-1,) + video_shape).transpose(0, 1).sum((1, 2))
            i2i = loss[1:].sum().mul_(video_shape[0] / (video_shape[0] - 1))
            return {"loss_t2i": loss[0].mul(video_shape[0]), "loss_i2i": i2i}
        return {"loss": loss.sum()}

    @torch.no_grad()
    def denoise(self, z, x, guidance_scaler, generator=None, pred_ids=None) -> torch.Tensor:
        """Run diffusion denoising process."""
        self.sample_scheduler._step_index = None  # Reset counter.
        for t in self.sample_scheduler.timesteps:
            z, pred_ids = guidance_scaler.maybe_disable(t, z, pred_ids)
            timestep = torch.as_tensor(t, device=x.device).expand(z.shape[0])
            model_pred = self.image_decoder(guidance_scaler.expand(x), timestep, z, pred_ids)
            model_pred = guidance_scaler.scale(model_pred)
            model_pred = self.image_encoder.patch_embed.unpatchify(model_pred)
            x = self.sample_scheduler.step(model_pred, t, x, generator=generator).prev_sample
        return self.image_encoder.patch_embed.patchify(x)

    @torch.inference_mode()
    def generate_frame(self, states: Dict, inputs: Dict):
        """Generate a batch of frames."""
        guidance_scaler = GuidanceScaler(**inputs)
        generator = self.mask_embed.generator = inputs.get("generator", None)
        all_num_preds = [_ for _ in inputs["num_preds"] if _ > 0]
        c, x, self.mask_embed.mask = states["c"], states["x"].zero_(), None
        pos = self.image_pos_embed.get_pos(1, c.size(0)) if self.image_pos_embed else None
        for i, num_preds in enumerate(self.progress_bar(all_num_preds, inputs.get("tqdm2", False))):
            guidance_scaler.decay_guidance_scale((i + 1) / len(all_num_preds))
            z = self.mask_embed(self.image_encoder.patch_embed(x))
            pred_mask, pred_ids = self.mask_embed.get_pred_mask(num_preds)
            pred_ids = guidance_scaler.expand(pred_ids)
            prev_ids = prev_ids if i else pred_ids.new_empty((pred_ids.size(0), 0, 1))
            z = self.image_encoder(guidance_scaler.expand(z), c, prev_ids, pos=pos)
            prev_ids = torch.cat([prev_ids, pred_ids], dim=1)
            states["noise"].normal_(generator=generator)
            sample = self.denoise(z, states["noise"], guidance_scaler.clone(), generator, pred_ids)
            x.add_(self.image_encoder.patch_embed.unpatchify(sample.mul_(pred_mask)))

    @torch.inference_mode()
    def generate_video(self, inputs: Dict):
        """Generate a batch of videos."""
        guidance_scaler = GuidanceScaler(**inputs)
        max_latent_length = inputs.get("max_latent_length", 1)
        self.sample_scheduler.set_timesteps(inputs.get("num_diffusion_steps", 25))
        states = {"x": inputs["x"], "noise": inputs["x"].clone()}
        latents, self.mask_embed.pred_ids, time_pos = inputs.get("latents", []), None, []
        if self.image_pos_embed:  # RoPE.
            time_pos = self.video_pos_embed.get_pos(max_latent_length).chunk(max_latent_length, 1)
        else:  # Absolute PE, which will be deprecated in the future.
            time_embed = self.video_pos_embed.get_time_embed(max_latent_length)
        inputs["c"] = guidance_scaler.expand_text(inputs["c"])
        self.video_encoder.enable_kvcache(max_latent_length > 1)
        for states["t"] in self.progress_bar(range(max_latent_length), inputs.get("tqdm1", True)):
            pos = time_pos[states["t"]] if time_pos else None
            c = self.video_encoder.patch_embed(states["x"])
            c.__setitem__(slice(None), self.mask_embed.bos_token) if states["t"] == 0 else c
            c = self.video_pos_embed(c.add_(time_embed[states["t"]])) if not time_pos else c
            c = guidance_scaler.expand(c, padding=self.mask_embed.bos_token)
            c = states["c"] = self.video_encoder(c, None if states["t"] else inputs["c"], pos=pos)
            if not isinstance(self.video_encoder.mixer, torch.nn.Identity):
                states["c"] = self.video_encoder.mixer(states["*"], c) if states["t"] else c
                states["*"] = states["*"] if states["t"] else states["c"]
            if states["t"] == 0 and latents:
                states["x"].copy_(latents[-1])
            else:
                self.generate_frame(states, inputs)
                latents.append(states["x"].clone())
        self.video_encoder.enable_kvcache(False)

    def train_video(self, inputs):
        """Train a batch of videos."""
        # 3D temporal autoregressive modeling (TAM).
        inputs["x"].unsqueeze_(2) if inputs["x"].dim() == 4 else None
        bs, latent_length = inputs["x"].size(0), inputs["x"].size(2)
        c = self.video_encoder.patch_embed(inputs["x"][:, :, : latent_length - 1])
        bov = self.mask_embed.bos_token.expand(bs, 1, c.size(-2), -1)
        c, pos = self.video_pos_embed(torch.cat([bov, c], dim=1)), None
        if self.image_pos_embed:
            pos = self.video_pos_embed.get_pos(c.size(1), bs, self.video_encoder.patch_embed.hw)
        attn_mask = self.mask_embed.get_attn_mask(c, inputs["c"]) if latent_length > 1 else None
        [setattr(blk.attn, "attn_mask", attn_mask) for blk in self.video_encoder.blocks]
        c = self.video_encoder(c.flatten(1, 2), inputs["c"], pos=pos)
        if not isinstance(self.video_encoder.mixer, torch.nn.Identity) and latent_length > 1:
            c = c.view(bs, latent_length, -1, c.size(-1)).split([1, latent_length - 1], 1)
            c = torch.cat([c[0], self.video_encoder.mixer(*c)], 1)
        # 2D masked autoregressive modeling (MAM).
        x = inputs["x"][:, :, :latent_length].transpose(1, 2).flatten(0, 1)
        z, bs = self.image_encoder.patch_embed(x), bs * latent_length
        if self.image_pos_embed:
            pos = self.image_pos_embed.get_pos(1, bs, self.image_encoder.patch_embed.hw)
        z = self.image_encoder(self.mask_embed(z), c.reshape(bs, -1, c.size(-1)), pos=pos)
        # 1D token-wise diffusion modeling (MLP).
        video_shape = (latent_length, z.size(1)) if latent_length > 1 else None
        return self.get_losses(z, x, video_shape=video_shape)

    def forward(self, inputs):
        """Define the computation performed at every call."""
        self.pipeline_preprocess(inputs)
        self.preprocess(inputs)
        if self.training:
            return self.train_video(inputs)
        inputs["latents"] = inputs.pop("latents", [])
        self.generate_video(inputs)
        return {"x": torch.stack(inputs["latents"], dim=2)}
