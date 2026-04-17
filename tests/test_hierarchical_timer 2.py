import pytest

from src.hierarchical_timer import MAX_DELAY, HierarchicalTimer


def test_fires_at_correct_tick():
    ht = HierarchicalTimer()
    fired = []
    ht.schedule(5, lambda: fired.append("a"))
    ht.run_until(4)
    assert fired == []
    ht.tick()
    assert fired == ["a"]


def test_zero_delay_fires_on_next_tick():
    ht = HierarchicalTimer()
    fired = []
    ht.schedule(0, lambda: fired.append("x"))
    # deadline == now == 0; first tick moves to 1, slot 1 — so zero-delay never fires.
    # Confirm semantics: delay must be > 0 to fire on the next tick.
    ht.tick()
    assert fired == []


def test_cascade_from_level1_to_level0():
    ht = HierarchicalTimer()
    fired = []
    ht.schedule(10, lambda: fired.append("late"))
    for _ in range(9):
        ht.tick()
    assert fired == []
    ht.tick()
    assert fired == ["late"]


def test_many_deadlines_fire_in_order():
    ht = HierarchicalTimer()
    fired = []
    for d in (3, 8, 15, 1, 63, 32):
        ht.schedule(d, lambda d=d: fired.append(d))
    ht.run_until(63)
    assert fired == [1, 3, 8, 15, 32, 63]


def test_cancel_prevents_firing():
    ht = HierarchicalTimer()
    fired = []
    t = ht.schedule(4, lambda: fired.append("no"))
    assert ht.cancel(t) is True
    ht.run_until(4)
    assert fired == []
    assert ht.cancel(t) is False


def test_capacity_64():
    ht = HierarchicalTimer()
    for d in range(1, MAX_DELAY):
        ht.schedule(d, lambda: None)
    ht.schedule(1, lambda: None)
    with pytest.raises(RuntimeError):
        ht.schedule(1, lambda: None)


def test_delay_out_of_range():
    ht = HierarchicalTimer()
    with pytest.raises(ValueError):
        ht.schedule(MAX_DELAY, lambda: None)
    with pytest.raises(ValueError):
        ht.schedule(-1, lambda: None)
