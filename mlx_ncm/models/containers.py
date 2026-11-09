"""Containers that reproduce ``torch.nn`` child naming.

``torch.nn.Sequential`` / ``torch.nn.ModuleList`` name their children ``"0"``,
``"1"``, ... whereas ``mlx.nn`` nests them under ``layers.<i>``. Checkpoint keys
must line up with PyTorch, so these containers store children under numeric
attribute names (MLX traverses them as ``<attr>.0.weight`` and supports
non-identifier keys).
"""

import mlx.nn as nn


class TorchIndexed(nn.Module):
    """Index-addressable module list with ``"0"``, ``"1"``, ... child names."""

    def __init__(self, mods):
        super().__init__()
        mods = list(mods)
        self._mods = tuple(mods)
        for i, m in enumerate(mods):
            setattr(self, str(i), m)

    def __len__(self):
        return len(self._mods)

    def __getitem__(self, i):
        return self._mods[i]

    def __iter__(self):
        return iter(self._mods)

    def __call__(self, x):
        for m in self._mods:
            x = m(x)
        return x


class Sequential(TorchIndexed):
    """``torch.nn.Sequential`` equivalent (varargs constructor)."""

    def __init__(self, *mods):
        super().__init__(list(mods))


class ModuleList(TorchIndexed):
    """``torch.nn.ModuleList`` equivalent (single-sequence constructor)."""

    def __init__(self, mods):
        super().__init__(list(mods))

