# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import json
from typing import Type, TypeVar, Any, Callable

from omegaconf import DictConfig, ListConfig, OmegaConf

from lingua.optim import SchedulerType
from lingua.transformer import InitStdFactor, RoPEType
from lingua.data import DatasetType
from lingua.tokenizer import TokenizerType

logger = logging.getLogger()

T = TypeVar("T")


def set_struct_recursively(cfg: DictConfig | ListConfig, strict: bool = True) -> None:
    # Set struct mode for the current level
    OmegaConf.set_struct(cfg, strict)

    # Traverse through nested dictionaries and lists
    if isinstance(cfg, DictConfig):
        for value in cfg.values():
            if isinstance(value, (DictConfig, ListConfig)):
                set_struct_recursively(value, strict)
    elif isinstance(cfg, ListConfig):
        for item in cfg:
            if isinstance(item, (DictConfig, ListConfig)):
                set_struct_recursively(item, strict)


def flatten_dict(dictionary: dict, parent_key="", sep="_"):
    items = []
    for key, value in dictionary.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


ENCODED_ENUMS = {
    SchedulerType,
    InitStdFactor,
    DatasetType,
    RoPEType,
    TokenizerType,
}

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in ENCODED_ENUMS:
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def enum_decoder(obj):
    if "__enum__" in obj:
        name, member = obj["__enum__"].split(".")

        enum_class = next((enum for enum in ENCODED_ENUMS if enum.__name__ == name), None)
        if enum_class is None:
            raise ValueError(f"Unknown enum class: {name}")

        return getattr(enum_class, member)
    return obj

def apply_object_hook(
    data: DictConfig | ListConfig | dict | list | Any,
    hook: Callable,
) -> DictConfig | ListConfig | Any:
    decoded_data = hook(data) if isinstance(data, (DictConfig, dict)) else data

    if isinstance(decoded_data, (DictConfig, dict)):
        return OmegaConf.create({k: apply_object_hook(v, hook) for k, v in decoded_data.items()})
    elif isinstance(decoded_data, (ListConfig, list)):
        return OmegaConf.create([apply_object_hook(item, hook) for item in decoded_data])
    else:
        return decoded_data

def dataclass_from_dict(cls: Type[T], data: dict, strict: bool = True) -> T:
    """
    Converts a dictionary to a dataclass instance, recursively for nested structures.
    """
    enum_decoded_data = apply_object_hook(data, enum_decoder)  # Decode enums if present

    base = OmegaConf.structured(cls())
    OmegaConf.set_struct(base, strict)
    override = OmegaConf.create(enum_decoded_data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))


def load_config_file(config_file, dataclass_cls: Type[T]) -> T:
    config = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)
    return dataclass_from_dict(dataclass_cls, config)


def dump_config(config, path, log_config=True):
    yaml_dump = OmegaConf.to_yaml(OmegaConf.structured(config))
    with open(path, "w") as f:
        if log_config:
            logger.info("Using the following config for this run:")
            logger.info(yaml_dump)
        f.write(yaml_dump)
