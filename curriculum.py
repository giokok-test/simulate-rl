"""Curriculum utilities for pursuit-evasion training.

This module exposes a small scheduler that interpolates environment
configuration between a ``start`` and an ``end`` dictionary.  Two modes are
supported:

* ``"fixed"`` – progress advances linearly with the episode index.
* ``"adaptive"`` – progress increases once the recent success rate exceeds
  ``success_threshold`` (Bengio et al., 2009).

The :func:`initialize_gym` helper applies the current curriculum state before
instantiating :class:`~pursuit_evasion.PursuerOnlyEnv` so that training and
evaluation share identical environment initialisation.

Example
-------
```python
from curriculum import Curriculum, initialize_gym
from pursuit_evasion import load_config

base_cfg = load_config()
cur = Curriculum(start={...}, end={...}, mode="adaptive", stages=10)
env = initialize_gym(base_cfg, curriculum=cur, max_steps=500)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import copy
from typing import Deque, Optional

from pursuit_evasion import apply_curriculum, PursuerOnlyEnv


@dataclass
class Curriculum:
    """Curriculum controller supporting fixed and adaptive schedules."""

    start: dict
    end: dict
    mode: str = "adaptive"
    stages: int = 2
    success_threshold: float = 0.6
    window: int = 64
    stage: int = 0
    _recent: Deque[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:  # noqa: D401
        self._recent = deque(maxlen=self.window)

    @property
    def progress(self) -> float:
        """Return interpolation progress in ``[0, 1]``."""
        return self.stage / max(self.stages - 1, 1)

    def advance(self, episode: int, total_episodes: int) -> None:
        """Advance stage for fixed curricula based on episode index."""
        if self.mode != "fixed":
            return
        num_transitions = max(self.stages - 1, 1)
        self.stage = (episode * num_transitions) // max(total_episodes - 1, 1)

    def update(self, success: bool) -> None:
        """Update stage for adaptive curricula based on success history."""
        if self.mode != "adaptive":
            return
        self._recent.append(1 if success else 0)
        if (
            len(self._recent) >= self.window
            and sum(self._recent) / len(self._recent) >= self.success_threshold
            and self.stage < self.stages - 1
        ):
            self.stage += 1
            self._recent.clear()

    def configure(self, base_cfg: dict) -> dict:
        """Return a copy of ``base_cfg`` with curriculum applied."""
        cfg = copy.deepcopy(base_cfg)
        apply_curriculum(cfg, self.start, self.end, self.progress)
        return cfg


def initialize_gym(
    base_cfg: dict,
    *,
    curriculum: Optional[Curriculum] = None,
    max_steps: Optional[int] = None,
    capture_bonus: float = 0.0,
) -> PursuerOnlyEnv:
    """Create a :class:`PursuerOnlyEnv` with optional curriculum."""
    cfg = base_cfg if curriculum is None else curriculum.configure(base_cfg)
    return PursuerOnlyEnv(cfg, max_steps=max_steps, capture_bonus=capture_bonus)

