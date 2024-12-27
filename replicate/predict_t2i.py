# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from cog import BasePredictor, Input, Path
import torch
from diffnext.pipelines import NOVAPipeline


MODEL_CACHE = "model_cache"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/BAAI/nova-d48w1536-sdxl1024/model_cache.tar"
)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            print("downloading")
            download_weights(MODEL_URL, MODEL_CACHE)

        model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
        self.pipe = NOVAPipeline.from_pretrained(
            f"{MODEL_CACHE}/BAAI/nova-d48w1536-sdxl1024", **model_args
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a shiba inu wearing a beret and black turtleneck.",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="low quality, deformed, distorted, disfigured, fused fingers, bad anatomy, weird hand",
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps", ge=1, le=128, default=64
        ),
        num_diffusion_steps: int = Input(
            description="Number of diffusion steps", ge=1, le=50, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=10, default=5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            num_diffusion_steps=num_diffusion_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        out_path = "/tmp/out.png"
        output.images[0].save(out_path)
        return Path(out_path)
