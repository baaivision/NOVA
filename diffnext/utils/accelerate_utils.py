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
"""Accelerate utilities."""

import atexit
import functools
import logging
import os
import sys
import time

import accelerate
import torch
import wandb

from diffnext.utils.omegaconf_utils import flatten_omega_conf


def build_accelerator(config, **kwargs) -> accelerate.Accelerator:
    """Build accelerator."""
    accelerator = accelerate.Accelerator(
        log_with=kwargs.get("log_with", None),
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )
    if hasattr(accelerator.state.deepspeed_plugin, "deepspeed_config"):
        import deepspeed

        deepspeed.logger.setLevel(kwargs.get("deepspeed_log_lvl", "WARNING"))
        # Dummy size to avoid the raised errors.
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
    return accelerator


def build_wandb(config, accelerator):
    """Build wandb for accelerator."""
    if "wandb" not in config or not accelerator.is_main_process:
        return
    config.wandb = config.wandb or type(config)({})
    old_run_id = config.wandb.get("run_id", None)
    config.wandb.run_id = run_id = old_run_id or wandb.util.generate_id()
    init_kwargs = dict(id=run_id, name=config.experiment.name, resume=old_run_id is not None)
    init_kwargs["config"] = wandb_config = {k: v for k, v in flatten_omega_conf(config, True)}
    accelerator.init_trackers(config.experiment.project, wandb_config, {"wandb": init_kwargs})


def get_ddp_shards(accelerator) -> dict:
    """Return the shard arguments for simple DDP."""
    return {"shard_id": accelerator.process_index, "num_shards": accelerator.num_processes}


def precision_to_dtype(precision="bf16") -> torch.dtype:
    """Convert precision string to torch dtype."""
    str_dict = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
    return getattr(torch, str_dict.get(precision.lower(), "float32"))


@functools.lru_cache()
def set_logger(output_dir=None, name="diffnext", level="INFO", accelerator=None):
    """Set logger."""

    @functools.lru_cache(maxsize=None)
    def cached_log_stream(filename):
        """Register a cached filename."""
        f = open(filename, "a")
        atexit.register(f.close)
        return f

    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level), ch.setFormatter(formatter), logger.addHandler(ch)
    output_dir = "" if (accelerator and not accelerator.is_main_process) else output_dir
    if output_dir:
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        log_file = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())) + ".log"
        fh = logging.StreamHandler(cached_log_stream(os.path.join(output_dir, "logs", log_file)))
        fh.setLevel(level), fh.setFormatter(formatter), logger.addHandler(fh)
    return accelerate.logging.MultiProcessAdapter(logger, {}) if accelerator else logger
