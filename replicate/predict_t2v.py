# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path
import torch
from diffnext.pipelines import NOVAPipeline
from diffnext.utils import export_to_video

# Fix tokenizer fork issue.
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Switch to the allocator optimized for dynamic shape.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


MODEL_CACHE = "model_cache"
MODEL_URL = f"https://weights.replicate.delivery/default/BAAI/nova-d48w1024-osp480/model_cache.tar"

VIDEO_PRESETS = {"label": "33x768x480", "w": 768, "h": 480, "#latents": 9}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def crop_image(image, target_h, target_w):
    """Center crop image to target size."""
    h, w = image.height, image.width
    aspect_ratio_target, aspect_ratio = target_w / target_h, w / h
    if aspect_ratio > aspect_ratio_target:
        new_w = int(h * aspect_ratio_target)
        x_start = (w - new_w) // 2
        image = image.crop((x_start, 0, x_start + new_w, h))
    else:
        new_h = int(w / aspect_ratio_target)
        y_start = (h - new_h) // 2
        image = image.crop((0, y_start, w, y_start + new_h))
    return np.array(image.resize((target_w, target_h), PIL.Image.Resampling.BILINEAR))


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            print("downloading")
            download_weights(MODEL_URL, MODEL_CACHE)

        model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
        self.pipe = NOVAPipeline.from_pretrained(
            f"{MODEL_CACHE}/BAAI/nova-d48w1024-osp480/", **model_args
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="The camera slowly rotates around a massive stack of vintage televisions that are placed within a large New York museum gallery. Each of the televisions is showing a different program. There are 1950s sci-fi movies with their distinctive visuals, horror movies with their creepy scenes, news broadcasts with moving images and words, static on some screens, and a 1970s sitcom with its characteristic look. The televisions are of various sizes and designs, some with rounded edges and others with more angular shapes. The gallery is well-lit, with light falling on the stack of televisions and highlighting the different programs being shown. There are no people visible in the immediate vicinity, only the stack of televisions and the surrounding gallery space.",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="low quality, deformed, distorted, disfigured, fused fingers, bad anatomy, weird hand",
        ),
        image: Path = Input(description="Input image prompt, optional", default=None),
        num_inference_steps: int = Input(
            description="Number of inference steps", ge=1, le=128, default=128
        ),
        num_diffusion_steps: int = Input(
            description="Number of diffusion steps", ge=1, le=100, default=100
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=10, default=7
        ),
        motion_flow: int = Input(description="Motion Flow", ge=1, le=10, default=5),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        fps: int = Input(description="fps for the output video", default=12),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = (
            crop_image(Image.open(str(image)), VIDEO_PRESETS["h"], VIDEO_PRESETS["w"])
            if image
            else None
        )
        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            motion_flow=motion_flow,
            preset=VIDEO_PRESETS,
            num_inference_steps=num_inference_steps,
            num_diffusion_steps=num_diffusion_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_latent_length=VIDEO_PRESETS["#latents"],
        )
        print(type(output))
        out_path = "/tmp/out.mp4"
        export_to_video(output.frames[0], out_path, fps=fps)
        return Path(out_path)
