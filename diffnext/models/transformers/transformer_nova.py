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
"""3D transformer model for AR generation in NOVA."""

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from diffnext.models.diffusion_mlp import DiffusionMLP
from diffnext.models.embeddings import PosEmbed, VideoPosEmbed, RotaryEmbed3D
from diffnext.models.embeddings import MaskEmbed, MotionEmbed, TextEmbed, LabelEmbed
from diffnext.models.normalization import AdaLayerNorm
from diffnext.models.transformers.transformer_3d import Transformer3DModel
from diffnext.models.vision_transformer import VisionTransformer
from diffnext.utils.registry import Registry

VIDEO_ENCODERS = Registry("video_encoders")
IMAGE_ENCODERS = Registry("image_encoders")
IMAGE_DECODERS = Registry("image_decoders")


@VIDEO_ENCODERS.register("vit_d16w768", depth=16, embed_dim=768, num_heads=12)
@VIDEO_ENCODERS.register("vit_d16w1024", depth=16, embed_dim=1024, num_heads=16)
@VIDEO_ENCODERS.register("vit_d16w1536", depth=16, embed_dim=1536, num_heads=16)
def video_encoder(depth, embed_dim, num_heads, patch_size, image_size, image_dim):
    return VisionTransformer(**locals())


@IMAGE_ENCODERS.register("vit_d32w768", depth=32, embed_dim=768, num_heads=12)
@IMAGE_ENCODERS.register("vit_d32w1024", depth=32, embed_dim=1024, num_heads=16)
@IMAGE_ENCODERS.register("vit_d32w1536", depth=32, embed_dim=1536, num_heads=16)
def image_encoder(depth, embed_dim, num_heads, patch_size, image_size, image_dim):
    return VisionTransformer(**locals())


@IMAGE_DECODERS.register("mlp_d3w1280", depth=3, embed_dim=1280)
@IMAGE_DECODERS.register("mlp_d6w768", depth=6, embed_dim=768)
@IMAGE_DECODERS.register("mlp_d6w1024", depth=6, embed_dim=1024)
@IMAGE_DECODERS.register("mlp_d6w1536", depth=6, embed_dim=1536)
def image_decoder(depth, embed_dim, patch_size, image_dim, cond_dim):
    return DiffusionMLP(**locals())


class NOVATransformer3DModel(Transformer3DModel, ModelMixin, ConfigMixin):
    """A 3D transformer model for AR generation in NOVA."""

    @register_to_config
    def __init__(
        self,
        image_dim=None,
        image_size=None,
        image_stride=None,
        text_token_dim=None,
        text_token_len=None,
        image_base_size=None,
        video_base_size=None,
        video_mixer_rank=None,
        rotary_pos_embed=False,
        arch=("", "", ""),
    ):
        image_size = (image_size,) * 2 if isinstance(image_size, int) else image_size
        image_size = tuple(v // image_stride for v in image_size)
        image_args = {"image_dim": image_dim, "patch_size": 15 // image_stride + 1}
        video_args = {**image_args, "patch_size": image_args["patch_size"] * 2}
        video_encoder = VIDEO_ENCODERS.get(arch[0])(image_size=image_size, **video_args)
        image_encoder = IMAGE_ENCODERS.get(arch[1])(image_size=image_size, **image_args)
        image_decoder = IMAGE_DECODERS.get(arch[2])(cond_dim=image_encoder.embed_dim, **image_args)
        if rotary_pos_embed:
            video_pos_embed = RotaryEmbed3D(video_encoder.rope.dim, video_base_size[1:])
            image_pos_embed = RotaryEmbed3D(image_encoder.rope.dim, image_base_size)
        else:
            video_pos_embed = VideoPosEmbed(video_encoder.embed_dim, video_base_size)
            image_encoder.pos_embed = PosEmbed(image_encoder.embed_dim, image_base_size)
        image_pos_embed = image_pos_embed if rotary_pos_embed else None
        if video_mixer_rank:
            video_mixer_rank = max(video_mixer_rank, 0)  # Use vanilla AdaLN if ``rank`` < 0.
            video_encoder.mixer = AdaLayerNorm(video_encoder.embed_dim, video_mixer_rank, eps=None)
        if text_token_dim:
            text_embed = TextEmbed(text_token_dim, image_encoder.embed_dim, text_token_len)
        super(NOVATransformer3DModel, self).__init__(
            video_encoder=video_encoder,
            image_encoder=image_encoder,
            image_decoder=image_decoder,
            mask_embed=MaskEmbed(image_encoder.embed_dim),
            text_embed=text_embed if text_token_dim else None,
            label_embed=LabelEmbed(image_encoder.embed_dim) if not text_token_dim else None,
            video_pos_embed=video_pos_embed,
            image_pos_embed=image_pos_embed,
            motion_embed=MotionEmbed(video_encoder.embed_dim) if video_base_size[0] > 1 else None,
        )
