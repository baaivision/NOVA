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
"""Flex data pipelines."""

import multiprocessing

import cv2
import numpy.random as npr

from diffnext.data import flex_transforms


class Worker(multiprocessing.Process):
    """Base data worker."""

    def __init__(self):
        super().__init__(daemon=True)
        self.seed = 1337
        self.reader_queue = None
        self.worker_queue = None

    def run(self):
        """Run implementation."""
        # Disable opencv threading and fix numpy random seed.
        cv2.setNumThreads(1), npr.seed(self.seed)
        while True:  # Main loop.
            self.worker_queue.put(self.get_outputs(self.reader_queue.get()))


class FeaturePipe(object):
    """Pipeline to transform data features."""

    def __init__(self):
        super().__init__()
        self.parse_latents = flex_transforms.ParseLatents()
        self.parse_annotations = flex_transforms.ParseAnnotations()

    def get_outputs(self, inputs):
        """Return the outputs."""
        latents = self.parse_latents(inputs)
        label, caption = self.parse_annotations(inputs)
        outputs = {"latents": [latents]}
        outputs.setdefault("prompt", [label]) if label is not None else None
        outputs.setdefault("prompt", [caption]) if caption is not None else None
        outputs.setdefault("motion_flow", [inputs["flow"]]) if "flow" in inputs else None
        return outputs


class FeatureWorker(FeaturePipe, Worker):
    """Worker to transform data features."""
