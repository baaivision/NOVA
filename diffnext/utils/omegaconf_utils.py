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
"""Omegaconf utilities."""

import importlib
import json
from typing import List

import omegaconf


class OmegaConfEncoder(json.JSONEncoder):
    """Custom JSON encoder for omegaconf objects."""

    def default(self, obj):
        if isinstance(obj, (omegaconf.ListConfig, omegaconf.DictConfig)):
            return omegaconf.OmegaConf.to_container(obj, resolve=True)
        return super().default(obj)


def get_config() -> omegaconf.DictConfig:
    """Return omega configurations from CLI."""
    cli_conf = omegaconf.OmegaConf.from_cli()
    omegaconf.OmegaConf.register_new_resolver("eval", eval)  # Register ``eval`` func.
    return omegaconf.OmegaConf.merge(omegaconf.OmegaConf.load(cli_conf.config), cli_conf)


def save_config(config: omegaconf.DictConfig, f):
    """Save config to YAML format string."""
    omegaconf.OmegaConf.save(config, f)


def config_to_yaml(config: omegaconf.DictConfig) -> str:
    """Dump config to YAML format string."""
    return omegaconf.OmegaConf.to_yaml(config)


def config_to_class(config: omegaconf.DictConfig) -> object:
    """Return the class object from config."""

    def get_obj_from_str(string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)

    if not config:
        return None
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])


def config_to_object(config: omegaconf.DictConfig, **kwargs) -> object:
    """Instantiate an object from config."""
    if not config:
        return None
    kwargs.update(config.get("params", dict()))
    return config_to_class(config)(**kwargs)


def flatten_omega_conf(cfg, resolve=True) -> List:
    """Flatten omega configurations."""
    ret = []

    def handle_dict(key, value, resolve):
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key, value, resolve):
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, omegaconf.DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, omegaconf.DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, omegaconf.ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, omegaconf.ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, omegaconf.DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, omegaconf.ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    return ret
