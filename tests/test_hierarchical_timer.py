import pytest

from src.hierarchical_timer import MAX_DELAY, HierarchicalTimer


def test_fires_at_correct_tick():
    ht = HierarchicalTimer()
    fired = []

    ht.schedule(5, lambda: fired.append("a"))

    assert len(ht) == 1
    ht.run_until(4)
    assert fired == []

    ht.tick()

    assert fired == ["a"]
    assert len(ht) == 0


def test_zero_delay_fires_on_next_tick():
    ht = HierarchicalTimer()
    fired = []

    ht.schedule(0, lambda: fired.append("x"))
    ht.tick()

    assert fired == ["x"]
    assert len(ht) == 0


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

    for delay in (3, 8, 15, 1, 63, 32):
        ht.schedule(delay, lambda delay=delay: fired.append(delay))

    ht.run_until(63)

    assert fired == [1, 3, 8, 15, 32, 63]


def test_cancel_prevents_firing():
    ht = HierarchicalTimer()
    fired = []
    timer = ht.schedule(4, lambda: fired.append("no"))

    assert ht.cancel(timer) is True
    assert len(ht) == 0

    ht.run_until(4)

    assert fired == []
    assert ht.cancel(timer) is False


def test_capacity_64():
    ht = HierarchicalTimer()

    for delay in range(1, MAX_DELAY):
        ht.schedule(delay, lambda: None)
    ht.schedule(1, lambda: None)

    with pytest.raises(RuntimeError):
        ht.schedule(1, lambda: None)


def test_delay_out_of_range():
    ht = HierarchicalTimer()

    with pytest.raises(ValueError):
        ht.schedule(MAX_DELAY, lambda: None)
    with pytest.raises(ValueError):
        ht.schedule(-1, lambda: None)


def test_cancelled_level1_timer_only_releases_one_slot():
    ht = HierarchicalTimer()
    timers = [ht.schedule(63, lambda: None) for _ in range(MAX_DELAY)]

    assert ht.cancel(timers[0]) is True
    assert len(ht) == MAX_DELAY - 1

    ht.run_until(56)
    ht.schedule(1, lambda: None)

    with pytest.raises(RuntimeError):
        ht.schedule(1, lambda: None)


def test_cancel_after_fire_returns_false():
    ht = HierarchicalTimer()
    fired = []
    timer = ht.schedule(1, lambda: fired.append("done"))

    ht.tick()

    assert fired == ["done"]
    assert ht.cancel(timer) is False
