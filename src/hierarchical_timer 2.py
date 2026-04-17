"""Hierarchical timing wheel for up to 64 concurrent timeouts.

Two-level wheel:
  - level 0: 8 slots, 1-tick granularity, covers deadlines within 7 ticks
  - level 1: 8 slots, 8-tick granularity, covers deadlines within 63 ticks

The wheel stores up to 64 pending timers and supports:
  - O(1) schedule
  - O(1) cancel via lazy tombstones
  - O(k) tick where k is the number of timers touched at that tick

`delay=0` is treated as "fire on the next tick" so callbacks always run from
`tick()` rather than synchronously inside `schedule()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Callable, Iterator

WHEEL_BITS = 3
WHEEL_SIZE = 1 << WHEEL_BITS          # 8
WHEEL_MASK = WHEEL_SIZE - 1           # 0b111
MAX_DELAY = WHEEL_SIZE * WHEEL_SIZE   # 64


@dataclass
class Timer:
    id: int
    deadline: int
    callback: Callable[[], None]
    cancelled: bool = False
    fired: bool = False

    @property
    def pending(self) -> bool:
        return not self.cancelled and not self.fired


@dataclass
class HierarchicalTimer:
    now: int = 0
    _ids: Iterator[int] = field(default_factory=lambda: count(1))
    _level0: list[list[Timer]] = field(
        default_factory=lambda: [[] for _ in range(WHEEL_SIZE)]
    )
    _level1: list[list[Timer]] = field(
        default_factory=lambda: [[] for _ in range(WHEEL_SIZE)]
    )
    _active: int = 0

    def __len__(self) -> int:
        return self._active

    def schedule(self, delay: int, callback: Callable[[], None]) -> Timer:
        if delay < 0 or delay >= MAX_DELAY:
            raise ValueError(f"delay must be in [0, {MAX_DELAY})")
        if self._active >= MAX_DELAY:
            raise RuntimeError(f"timer capacity {MAX_DELAY} exhausted")

        effective_delay = max(1, delay)
        timer = Timer(
            id=next(self._ids),
            deadline=self.now + effective_delay,
            callback=callback,
        )
        self._insert(timer)
        self._active += 1
        return timer

    def cancel(self, timer: Timer) -> bool:
        if not timer.pending:
            return False
        timer.cancelled = True
        self._active -= 1
        return True

    def tick(self) -> int:
        """Advance clock by one tick; fire due timers. Returns count fired."""
        self.now += 1
        slot0 = self.now & WHEEL_MASK

        if slot0 == 0:
            self._cascade()

        bucket = self._level0[slot0]
        self._level0[slot0] = []
        fired = 0
        for t in bucket:
            if not t.pending:
                continue
            t.fired = True
            self._active -= 1
            t.callback()
            fired += 1
        return fired

    def run_until(self, target_tick: int) -> int:
        total = 0
        while self.now < target_tick:
            total += self.tick()
        return total

    def _insert(self, t: Timer) -> None:
        remaining = t.deadline - self.now
        if remaining < WHEEL_SIZE:
            self._level0[t.deadline & WHEEL_MASK].append(t)
        else:
            self._level1[(t.deadline >> WHEEL_BITS) & WHEEL_MASK].append(t)

    def _cascade(self) -> None:
        slot1 = (self.now >> WHEEL_BITS) & WHEEL_MASK
        bucket = self._level1[slot1]
        self._level1[slot1] = []
        for t in bucket:
            if not t.pending:
                continue
            self._level0[t.deadline & WHEEL_MASK].append(t)
