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
"""Flex data transforms."""

import re
import numpy as np
import numpy.random as npr


class Transform(object):
    """Base transform type."""

    def filter_outputs(self, *outputs):
        outputs = [x for x in outputs if x is not None]
        return outputs if len(outputs) > 1 else outputs[0]


class ParseLatents(Transform):
    """Parse VQ or VAE latents."""

    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        for k, dtype in zip(("moments", "codes"), ("float16", "int32")):
            if k in inputs:
                return np.frombuffer(inputs[k], dtype).reshape(inputs["shape"])
        raise ValueError("Missing latents in inputs.")


class ParseAnnotations(Transform):
    """Parse ground-truth annotations."""

    def __init__(self, short_prob=0.5):
        super().__init__()
        self.short_prob = short_prob

    def __call__(self, inputs):
        text = inputs.get("text", None)
        label = inputs.get("label", None)
        caption = inputs.get("caption", None)
        if caption and isinstance(caption, dict):  # Cached.
            caption = np.frombuffer(caption["data"], "float16").reshape(caption["shape"])
            if text and isinstance(text, dict) and len(text["data"]) > 0 and npr.rand() < 0.5:
                caption = np.frombuffer(text["data"], "float16").reshape(text["shape"])
            return label, caption

        # Improved short caption.
        if label is None:
            text_match = re.match(r"^(.*?[.!?])\s+", caption)
            text = text if text else (text_match.group(1) if text_match else caption)
            caption = text if text and npr.rand() < self.short_prob else caption
        return label, caption
