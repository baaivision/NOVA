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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, esither express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Custom trainer focused on data parallelism specialization."""

import collections
import os
import shutil

import torch

from diffnext.engine import engine_utils
from diffnext.engine.model_ema import ModelEMA
from diffnext.pipelines.builder import build_pipeline
from diffnext.pipelines.builder import get_pipeline_path
from diffnext.utils import accelerate_utils
from diffnext.utils import profiler
from diffnext.utils.omegaconf_utils import config_to_class
from diffnext.utils.omegaconf_utils import config_to_object


class Trainer(object):
    """Schedule the iterative model training."""

    def __init__(self, config, accelerator, logger):
        """Create a trainer instance."""
        self.config, self.accelerator, self.logger = config, accelerator, logger
        self.dtype = accelerate_utils.precision_to_dtype(config.training.mixed_precision)
        self.train_dataloader = config_to_object(config.train_dataloader)
        self.pipe_path = get_pipeline_path(**config.pipeline.paths)
        self.pipe = build_pipeline(self.pipe_path, config_to_class(config.pipeline), self.dtype)
        self.pipe = self.pipe.to(device=engine_utils.get_device(config.training.gpu_id))
        self.ema = ModelEMA(self.pipe.model, **config.ema.params) if "ema" in config else None
        self.model = self.pipe.configure_model(config)
        param_groups = engine_utils.get_param_groups(self.model)
        self.optimizer = config_to_object(config.optimizer, params=param_groups)
        self.scheduler = config_to_object(config.lr_scheduler)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.metrics = collections.OrderedDict()
        if self.ema and config.experiment.resume_iter > 0:
            ckpt = config.experiment.resume_from_checkpoint
            ema_ckpt = ckpt.replace("checkpoints", "ema_checkpoints")
            ema_weights = os.path.join(ema_ckpt, config.model.name, "diffusion_pytorch_model.bin")
            engine_utils.load_weights(self.ema.model, ema_weights)

    @property
    def global_step(self) -> int:
        """Return the global iteration step.

        Returns:
            int: The global step.
        """
        return self.scheduler._step_count

    def save(self):
        """Save the checkpoint of current iterative step."""
        f = "checkpoint-{}/{}".format(self.global_step, self.config.model.name)
        f = os.path.join(self.config.experiment.output_dir, "checkpoints", f)
        if self.accelerator.is_main_process and not os.path.exists(f):
            self.model.save_pretrained(f, safe_serialization=False)
            self.logger.info("Wrote snapshot to: {:s}".format(f))
            if self.ema is not None:
                config_json = os.path.join(f, "config.json")
                f = f.replace("checkpoints", "ema_checkpoints")
                os.makedirs(f), shutil.copy(config_json, os.path.join(f, "config.json"))
                f = os.path.join(f, "diffusion_pytorch_model.bin")
                torch.save(self.ema.model.state_dict(), f)

    def add_metrics(self, stats):
        """Add or update the metrics.

        Args:
            stats (Dict)
                The current iteration stats.
        """
        for k, v in stats["metrics"].items():
            if k not in self.metrics:
                self.metrics[k] = profiler.SmoothedValue()
            self.metrics[k].update(v)

    def log_metrics(self, stats):
        """Send metrics to available trackers.

        Args:
            stats (Dict)
                The current iteration stats.
        """
        iter_template = "Iteration %d, lr = %.8f, time = %.2fs"
        metric_template = " " * 4 + "Train net output({}): {}"
        [self.logger.info(iter_template % (stats["step"], stats["lr"], stats["time"]))]
        [self.logger.info(metric_template.format(k, v)) for k, v in self.metrics.items()]
        tracker_logs = dict((k, v.median) for k, v in self.metrics.items())
        tracker_logs.update({"lr": stats["lr"], "time": stats["time"]})
        self.accelerator.log(tracker_logs, step=stats["step"])
        self.metrics.clear()

    def run_model(self, metrics, accum_steps=1):
        """Run multiple model steps.

        Args:
            metrics (Dict)
                The current iteration metrics.
            accum_step (int)
                The gradient accumulation steps.
        """
        for _ in range(accum_steps):
            inputs = self.train_dataloader.next()[0]
            outputs, losses = self.model(inputs), []
            for k, v in outputs.items():
                if "loss" not in k and "metric" not in k:
                    continue
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    losses.append(v)
                metrics[k] += float(self.accelerator.gather(v).mean()) / accum_steps
            losses = sum(losses[1:], losses[0])
            self.accelerator.accumulate().__enter__()
            self.accelerator.backward(losses)

    def run_step(self, accum_steps=1) -> dict:
        """Run single iteration step.

        Args:
            accum_step (int)
                The gradient accumulation steps.

        Returns:
            Dict: The current iteration stats.
        """
        stats = {"step": self.global_step}
        metrics = collections.defaultdict(float)
        timer = profiler.Timer().tic()
        stats["lr"] = self.scheduler.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = stats["lr"] * group.get("lr_scale", 1.0)
        self.run_model(metrics, accum_steps)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        stats["time"] = timer.toc()
        stats["metrics"] = collections.OrderedDict(sorted(metrics.items()))
        return stats

    def train_loop(self):
        """Training loop."""
        timer = profiler.Timer()
        max_steps = self.config.training.max_train_steps
        accum_steps = self.config.training.gradient_accumulation_steps
        log_every = self.config.experiment.log_every
        save_every = self.config.experiment.save_every
        self.scheduler._step_count = self.config.experiment.get("resume_iter", 0)
        while self.global_step < max_steps:
            with timer.tic_and_toc():
                stats = self.run_step(accum_steps)
            self.add_metrics(stats)
            if stats["step"] % log_every == 0:
                self.log_metrics(stats)
            if self.global_step % (10 * log_every) == 0:
                self.logger.info(profiler.get_progress(timer, self.global_step, max_steps))
            if self.ema and self.global_step % self.ema.update_every == 0:
                self.ema.update(self.model)
            if self.global_step % save_every == 0:
                self.save()
