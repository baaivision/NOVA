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
"""Simple implementation of continuous flow matching schedulers."""

import dataclasses
import math

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclasses.dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """Output for scheduler's `step` function output."""

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):

    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False):
        timesteps = np.arange(1, num_train_timesteps + 1, dtype="float32")[::-1]
        sigmas, self._shift = timesteps / num_train_timesteps, shift
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = torch.as_tensor(sigmas * num_train_timesteps)
        self.sigmas = torch.as_tensor(sigmas)
        self.sigma_min, self.sigma_max = float(sigmas[-1]), float(sigmas[0])
        self.timestep = self.sigma = None  # Training states.
        self._begin_index = self._step_index = None  # Inference counters.

    @property
    def shift(self):
        """The value used for shifting."""
        return self._shift

    @property
    def step_index(self):
        """The index counter for current timestep."""
        return self._step_index

    @property
    def begin_index(self):
        """The index for the first timestep."""
        return self._begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_shift(self, shift: float):
        self._shift = shift

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        return indices[1 if len(indices) > 1 else 0].item()

    def sample_timesteps(self, size, device=None):
        """Sample the discrete timesteps used for training."""
        dist = torch.normal(0, 1, size, device=device).sigmoid_()
        return dist.mul_(self.config.num_train_timesteps).to(dtype=torch.int64)

    def set_timesteps(self, num_inference_steps, mu=None):
        """Sets the discrete timesteps used for the diffusion chain."""
        self.num_inference_steps = num_inference_steps
        t_max, t_min = self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min)
        timesteps = np.linspace(t_max, t_min, num_inference_steps, dtype="float32")
        sigmas = timesteps / self.config.num_train_timesteps
        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.sigmas = sigmas.tolist() + [0]
        self.timesteps = sigmas * self.config.num_train_timesteps
        self._begin_index = self._step_index = None

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        """Add forward noise to samples for training."""
        dtype, device = original_samples.dtype, original_samples.device
        self.timestep = self.timesteps.to(device=device)[timesteps]
        self.sigma = self.sigmas.to(device=device, dtype=dtype)[timesteps]
        self.sigma = self.sigma.view(timesteps.shape + (1,) * (noise.dim() - timesteps.dim()))
        return self.sigma * noise + (1.0 - self.sigma) * original_samples

    def scale_noise(self, sample: torch.Tensor, timestep: float, noise: torch.Tensor):
        """Add forward noise to samples for inference."""
        self._init_step_index(timestep) if self.step_index is None else None
        sigma = self.sigmas[self.step_index]
        return sigma * noise + (1.0 - sigma) * sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.FloatTensor,
        generator: torch.Generator = None,
        return_dict=True,
    ):
        """Predict the sample from the previous timestep."""
        self._init_step_index(timestep) if self.step_index is None else None
        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        prev_sample = model_output.mul(dt).add_(sample)
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
