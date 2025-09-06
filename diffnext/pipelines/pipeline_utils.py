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
"""Pipeline utilities."""

from typing import List, Union

from diffusers.utils import BaseOutput
import numpy as np
import PIL.Image
import torch


class NOVAPipelineOutput(BaseOutput):
    """Output class for NOVA pipelines.

    Args:
        images (List[PIL.Image.Image] or np.ndarray)
            List of PIL images or numpy array of shape `(batch_size, height, width, num_channels)`.
        frames (np.ndarray)
            List of video frames. The array shape is `(batch_size, num_frames, height, width, num_channels)`
    """  # noqa

    images: Union[List[PIL.Image.Image], np.ndarray]
    frames: np.array


class PipelineMixin(object):
    """Base class for diffusion pipeline."""

    def register_module(self, model_or_path, name) -> torch.nn.Module:
        """Register pipeline component.

        Args:
            model_or_path (str or torch.nn.Module):
                The model or path to model.
            name (str):
                The module name.

        Returns:
            torch.nn.Module: The registered module.

        """
        model = model_or_path
        if isinstance(model_or_path, str):
            cls = self.__init__.__annotations__[name]
            if hasattr(cls, "from_pretrained") and model_or_path:
                model = cls.from_pretrained(model_or_path, torch_dtype=self.dtype)
                model = model.to(self.device) if isinstance(model, torch.nn.Module) else model
            model = cls()
        self.register_to_config(**{name: model.__class__.__name__})
        return model
