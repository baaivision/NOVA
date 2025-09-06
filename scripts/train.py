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
"""Train a diffnext model."""

import json
import os

from diffnext.engine.train_engine import Trainer
from diffnext.engine.train_engine import engine_utils
from diffnext.utils import accelerate_utils
from diffnext.utils import omegaconf_utils


def prepare_checkpoints(config):
    """Prepare checkpoints for model resuming.

    Args:
        config (omegaconf.DictConfig)
            The model config.
    """
    config.experiment.setdefault("resume_from_checkpoint", "")
    ckpt_dir = os.path.abspath(os.path.join(config.experiment.output_dir, "checkpoints"))
    resume_iter, _ = 0, os.makedirs(ckpt_dir, exist_ok=True)
    if config.experiment.resume_from_checkpoint == "latest":
        ckpts = [_ for _ in os.listdir(ckpt_dir) if _.startswith("checkpoint-")]
        if ckpts:
            resume_iter, ckpt = sorted((int(_.split("-")[-1]), _) for _ in ckpts)[-1]
            config.experiment.resume_from_checkpoint = os.path.join(ckpt_dir, ckpt)
    elif config.experiment.resume_from_checkpoint:
        resume_iter = int(os.path.split(config.experiment.resume_from_checkpoint).split("-")[-1])
    config.experiment.resume_iter = resume_iter
    if resume_iter:  # Override the pretrained path specified in the config.
        config.pipeline.paths.pretrained_path = config.experiment.resume_from_checkpoint


def prepare_datasets(config, accelerator):
    """Prepare datasets for model training.

    Args:
        config (omegaconf.DictConfig)
            The model config.
        accelerator (accelerate.Accelerator)
            The accelerator instance.
    """
    dataset = config.train_dataloader.params.dataset
    if os.path.exists(os.path.join(dataset, "METADATA")):
        with open(os.path.join(dataset, "METADATA"), "r") as f:
            max_examples = json.load(f)["entries"]
    else:
        raise ValueError("Unsupported dataset: " + dataset)
    config.train_dataloader.params.max_examples = max_examples
    if "shard_id" not in config.train_dataloader.params:
        # By default, we use dataset shards across all processes.
        config.train_dataloader.params.update(accelerate_utils.get_ddp_shards(accelerator))


def run_train(config, accelerator, logger):
    """Start a model training task.

    Args:
        config (omegaconf.DictConfig)
            The model config.
        accelerator (accelerate.Accelerator)
            The accelerator instance.
        logger (logging.Logger)
            The logger instance.
    """
    trainer = Trainer(config, accelerator, logger)
    logger.info("#Params: %.2fM" % engine_utils.count_params(trainer.model))
    logger.info("Start training...")
    trainer.train_loop()
    trainer.ema.update(trainer.model) if trainer.ema else None
    trainer.save()


def main():
    """Main entry point."""
    config = omegaconf_utils.get_config()
    accelerator = accelerate_utils.build_accelerator(config, log_with="wandb")
    accelerate_utils.build_wandb(config, accelerator=accelerator)
    logger = accelerate_utils.set_logger(config.experiment.output_dir, accelerator=accelerator)
    device_seed = config.training.seed + accelerator.process_index
    config.training.gpu_id, config.training.seed = accelerator.device.index, device_seed
    engine_utils.manual_seed(config.training.seed, (config.training.gpu_id, device_seed))
    prepare_checkpoints(config), prepare_datasets(config, accelerator)
    logger.info(f"Config:\n{omegaconf_utils.config_to_yaml(config)}")
    if accelerator.is_main_process:
        config_path = os.path.join(config.experiment.output_dir, "config.yaml")
        omegaconf_utils.save_config(config, config_path)
    run_train(config, accelerator, logger)


if __name__ == "__main__":
    main()
