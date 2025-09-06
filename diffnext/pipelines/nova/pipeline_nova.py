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
"""Non-quantized video autoregressive pipeline for NOVA."""

from typing import List

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import numpy as np
import torch

from diffnext.image_processor import VaeImageProcessor
from diffnext.pipelines.pipeline_utils import NOVAPipelineOutput, PipelineMixin


class NOVAPipeline(DiffusionPipeline, PipelineMixin):
    """NOVA autoregressive diffusion pipeline."""

    _optional_components = ["transformer", "scheduler", "vae", "text_encoder", "tokenizer"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        transformer=None,
        scheduler=None,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        trust_remote_code=True,
    ):
        super(NOVAPipeline, self).__init__()
        self.vae = self.register_module(vae, "vae")
        self.text_encoder = self.register_module(text_encoder, "text_encoder")
        self.tokenizer = self.register_module(tokenizer, "tokenizer")
        self.transformer = self.register_module(transformer, "transformer")
        self.scheduler = self.register_module(scheduler, "scheduler")
        self.transformer.sample_scheduler, self.guidance_scale = self.scheduler, 5.0
        if self.transformer.text_embed:
            self.tokenizer_max_length = self.transformer.text_embed.num_tokens
            self.transformer.text_embed.encoders = [self.tokenizer, self.text_encoder]
        self.image_processor = VaeImageProcessor()

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        num_inference_steps=64,
        num_diffusion_steps=25,
        max_latent_length=1,
        guidance_scale=5,
        guidance_trunc=0,
        guidance_renorm=1,
        image_guidance_scale=0,
        spatiotemporal_guidance_scale=0,
        flow_shift=None,
        motion_flow=5,
        negative_prompt=None,
        image=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        disable_progress_bar=False,
        output_type="pil",
        **kwargs,
    ) -> NOVAPipelineOutput:
        """The call function to the pipeline for generation.

        Args:
            prompt (str or List[str], *optional*):
                The prompt to be encoded.
            num_inference_steps (int, *optional*, defaults to 64):
                The number of autoregressive steps.
            num_diffusion_steps (int, *optional*, defaults to 25):
                The number of denoising steps.
            max_latent_length (int, *optional*, defaults to 1):
                The maximum number of latents to generate. ``1`` for image generation.
            guidance_scale (float, *optional*, defaults to 5):
                The classifier guidance scale.
            guidance_trunc (float, *optional*, defaults to 0):
                The truncation threshold to classifier guidance.
            guidance_renorm (float, *optional*, defaults to 1):
                The minimal renorm scale to classifier guidance.
            image_guidance_scale (float, *optional*, defaults to 0):
                The image guidance scale.
            spatiotemporal_guidance_scale (float, *optional*, defaults to 0):
                The spatiotemporal guidance scale.
            flow_shift (float, *optional*)
                The shift value for the timestep schedule.
            motion_flow  (float, *optional*, defaults to 5):
                The motion flow value for video generation.
            negative_prompt (str or List[str], *optional*):
                The prompt or prompts to guide what to not include in image generation.
            image (numpy.ndarray, *optional*):
                The image to be encoded.
            num_images_per_prompt (int, *optional*, defaults to 1):
                The number of images that should be generated per prompt.
            generator (torch.Generator, *optional*):
                The random generator.
            latents (List[torch.Tensor], *optional*)
                A list of prefilled VAE latents.
            prompt_embeds (List[torch.Tensor], *optional*)
                A list of precomputed prompt embeddings.
            negative_prompt_embeds (List[torch.Tensor], *optional*)
                A list of precomputed negative prompt embeddings.
            disable_progress_bar (bool, *optional*)
                Whether to disable all progress bars.
            output_type (str, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.

        Returns:
            NOVAPipelineOutput: The pipeline output.
        """
        self.guidance_scale = guidance_scale
        self.scheduler.set_shift(flow_shift) if flow_shift else None
        inputs = {"generator": generator, **locals()}
        num_patches = int(np.prod(self.transformer.config.image_base_size))
        mask_ratios = np.cos(0.5 * np.pi * np.arange(num_inference_steps + 1) / num_inference_steps)
        mask_length = np.round(mask_ratios * num_patches).astype("int64")
        inputs["num_preds"] = mask_length[:-1] - mask_length[1:]
        inputs["tqdm1"] = max_latent_length > 1 and not disable_progress_bar
        inputs["tqdm2"] = max_latent_length == 1 and not disable_progress_bar
        inputs["prompt"] = self.encode_prompt(**dict(_ for _ in inputs.items() if "prompt" in _[0]))
        inputs["latents"] = self.prepare_latents(image, num_images_per_prompt, generator, latents)
        inputs["batch_size"] = len(inputs["prompt"]) // (2 if guidance_scale > 1 else 1)
        inputs["motion_flow"] = [motion_flow] * inputs["batch_size"]
        _, outputs = inputs.pop("self"), self.transformer(inputs)
        if output_type != "latent":
            outputs["x"] = self.image_processor.decode_latents(self.vae, outputs["x"])
        output_name = {4: "images", 5: "frames"}[len(outputs["x"].shape)]
        outputs["x"] = self.image_processor.postprocess(outputs["x"], output_type)
        return NOVAPipelineOutput(**{output_name: outputs["x"]})

    def prepare_latents(
        self,
        image=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
    ) -> List[torch.Tensor]:
        """Prepare the video latents.

        Args:
            image (numpy.ndarray, *optional*):
                The image to be encoded.
            num_images_per_prompt (int, *optional*, defaults to 1):
                The number of images that should be generated per prompt.
            generator (torch.Generator, *optional*):
                The random generator.
            latents (List[torch.Tensor], *optional*)
                A list of prefilled VAE latents.

        Returns:
            List[torch.Tensor]: The encoded latents.
        """
        if latents is not None:
            return latents
        latents = []
        if image is not None:
            latents.append(self.encode_image(image, num_images_per_prompt, generator))
        return latents

    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt=1,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ) -> torch.Tensor:
        """Encode text prompts.

        Args:
            prompt (str or List[str], *optional*):
                The prompt to be encoded.
            num_images_per_prompt (int, *optional*, defaults to 1):
                The number of images that should be generated per prompt.
            negative_prompt (str or List[str], *optional*):
                The prompt or prompts not to guide the image generation.
            prompt_embeds (List[torch.Tensor], *optional*)
                A list of precomputed prompt embeddings.
            negative_prompt_embeds (List[torch.Tensor], *optional*)
                A list of precomputed negative prompt embeddings.

        Returns:
            torch.Tensor: The prompt embedding.
        """

        def select_or_pad(a, b, n=1):
            return [a or b] * n if isinstance(a or b, str) else (a or b)

        embedder = self.transformer.text_embed
        if prompt_embeds is not None:
            prompt_embeds = embedder.encode_prompts(prompt_embeds)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = embedder.encode_prompts(negative_prompt_embeds)
        if prompt_embeds is not None:
            if negative_prompt_embeds is None and self.guidance_scale > 1:
                bs, seqlen = prompt_embeds.shape[:2]
                negative_prompt_embeds = embedder.weight[:seqlen].expand(bs, -1, -1)
            if self.guidance_scale > 1:
                c = torch.cat([prompt_embeds, negative_prompt_embeds])
            return c.repeat_interleave(num_images_per_prompt, dim=0)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        negative_prompt = select_or_pad(negative_prompt, "", len(prompt))
        prompts = prompt + (negative_prompt if self.guidance_scale > 1 else [])
        c = embedder.encode_prompts(prompts)
        return c.repeat_interleave(num_images_per_prompt, dim=0)

    def encode_image(self, image, num_images_per_prompt=1, generator=None) -> torch.Tensor:
        """Encode image prompt.

        Args:
            image (numpy.ndarray):
                The image to be encoded.
            num_images_per_prompt (int):
                The number of images that should be generated per prompt.
            generator (torch.Generator, *optional*):
                The random generator.

        Returns:
            torch.Tensor: The image embedding.
        """
        x = torch.as_tensor(image, device=self.device).to(dtype=self.dtype)
        x = x.sub(127.5).div_(127.5).permute(2, 0, 1).unsqueeze_(0)
        x = self.vae.scale_(self.vae.encode(x).latent_dist.sample(generator))
        return x.expand(num_images_per_prompt, -1, -1, -1)
