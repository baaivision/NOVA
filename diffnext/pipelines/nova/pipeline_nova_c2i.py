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
"""Non-quantized autoregressive pipeline for NOVA."""

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import numpy as np
import torch

from diffnext.image_processor import VaeImageProcessor
from diffnext.pipelines.pipeline_utils import NOVAPipelineOutput, PipelineMixin


class NOVAC2IPipeline(DiffusionPipeline, PipelineMixin):
    """NOVA autoregressive diffusion pipeline."""

    _optional_components = ["transformer", "scheduler", "vae"]

    def __init__(self, transformer=None, scheduler=None, vae=None, trust_remote_code=True):
        super(NOVAC2IPipeline, self).__init__()
        self.vae = self.register_module(vae, "vae")
        self.transformer = self.register_module(transformer, "transformer")
        self.scheduler = self.register_module(scheduler, "scheduler")
        self.transformer.sample_scheduler, self.guidance_scale = self.scheduler, 5.0
        self.image_processor = VaeImageProcessor()

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        num_inference_steps=64,
        num_diffusion_steps=25,
        guidance_scale=5,
        min_guidance_scale=None,
        negative_prompt=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        disable_progress_bar=False,
        output_type="pil",
        **kwargs,
    ) -> NOVAPipelineOutput:
        """The call function to the pipeline for generation.

        Args:
            prompt (int or List[int], *optional*):
                The prompt to be encoded.
            num_inference_steps (int, *optional*, defaults to 64):
                The number of autoregressive steps.
            num_diffusion_steps (int, *optional*, defaults to 25):
                The number of denoising steps.
            guidance_scale (float, *optional*, defaults to 5):
                The classifier guidance scale.
            min_guidance_scale (float, *optional*):
                The minimum classifier guidance scale.
            negative_prompt (int or List[int], *optional*):
                The prompt or prompts to guide what to not include in image generation.
            num_images_per_prompt (int, *optional*, defaults to 1):
                The number of images that should be generated per prompt.
            generator (torch.Generator, *optional*):
                The random generator.
            disable_progress_bar (bool, *optional*)
                Whether to disable all progress bars.
            output_type (str, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.

        Returns:
            NOVAPipelineOutput: The pipeline output.
        """
        self.guidance_scale = guidance_scale
        inputs = {"generator": generator, **locals()}
        num_patches = int(np.prod(self.transformer.config.image_base_size))
        mask_ratios = np.cos(0.5 * np.pi * np.arange(num_inference_steps + 1) / num_inference_steps)
        mask_length = np.round(mask_ratios * num_patches).astype("int64")
        inputs["num_preds"] = mask_length[:-1] - mask_length[1:]
        inputs["tqdm1"], inputs["tqdm2"], inputs["latents"] = False, not disable_progress_bar, []
        inputs["c"] = [self.encode_prompt(**dict(_ for _ in inputs.items() if "prompt" in _[0]))]
        inputs["batch_size"] = len(inputs["c"][0]) // (2 if guidance_scale > 1 else 1)
        _, outputs = inputs.pop("self"), self.transformer(inputs)
        if output_type != "latent":
            outputs["x"] = self.image_processor.decode_latents(self.vae, outputs["x"])
        outputs["x"] = self.image_processor.postprocess(outputs["x"], output_type)
        return NOVAPipelineOutput(**{"images": outputs["x"]})

    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt=1,
        negative_prompt=None,
    ) -> torch.Tensor:
        """Encode class prompts.

        Args:
            prompt (int or List[int], *optional*):
                The prompt to be encoded.
            num_images_per_prompt (int, *optional*, defaults to 1):
                The number of images that should be generated per prompt.
            negative_prompt (int or List[int], *optional*):
                The prompt or prompts to guide what to not include in image generation.

        Returns:
            torch.Tensor: The prompt index.
        """

        def select_or_pad(a, b, n=1):
            return [a or b] * n if isinstance(a or b, int) else (a or b)

        num_classes = self.transformer.label_embed.num_classes
        prompt = [prompt] if isinstance(prompt, int) else prompt
        negative_prompt = select_or_pad(negative_prompt, num_classes, len(prompt))
        prompts = prompt + (negative_prompt if self.guidance_scale > 1 else [])
        c = self.transformer.label_embed(torch.as_tensor(prompts, device=self.device))
        return c.repeat_interleave(num_images_per_prompt, dim=0)
