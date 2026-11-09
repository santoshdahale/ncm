"""Small helpers shared by the MLX port (mirrors ``utils/misc.py``)."""

import importlib
import math


def count_params(model, exclude_params=[]):
    """(total, non-embedding) parameter counts, matching ``utils.misc.count_params``."""
    from .ops import flatten_params

    flat = flatten_params(model)
    n_params = sum(int(v.size) for v in flat.values())

    non_embd_params = n_params
    if exclude_params:
        for name, value in flat.items():
            if any(exclude_param in name for exclude_param in exclude_params):
                non_embd_params -= int(value.size)

    return n_params, non_embd_params


def import_class(dotted_path):
    """Replace ``hydra.utils.get_class`` (the repo only ever needs a dotted path)."""
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"{dotted_path!r} is not a dotted path")
    return getattr(importlib.import_module(module_path), class_name)


def round_levels(levels, start, end):
    rounded_levels = [start]
    for i in range(1, len(levels) - 1):
        if math.floor(levels[i]) > rounded_levels[i - 1] and math.ceil(levels[i]) < end:
            rounded_levels.append(math.floor(levels[i]))
        elif (math.floor(levels[i]) <= rounded_levels[i - 1]
              and math.ceil(levels[i]) > rounded_levels[i - 1]
              and math.ceil(levels[i]) < end):
            rounded_levels.append(math.ceil(levels[i]))
        else:
            rounded_levels.append(rounded_levels[i - 1] + 1)

    rounded_levels.append(end)

    return rounded_levels
