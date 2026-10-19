from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from cftime import datetime as cftime_datetime
from hypothesis import given

import parcels._strategies as pst  # Parcels strategies
from parcels._core.utils.time import (
    TimeInterval,
    _get_cf_attrs,
    is_compatible,
    maybe_convert_python_timedelta_to_numpy,
    timedelta_to_float,
)


@pytest.mark.parametrize(
    "left,right",
    [
        (cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 2, calendar="gregorian")),
        (cftime_datetime(2023, 6, 1, calendar="365_day"), cftime_datetime(2023, 6, 2, calendar="365_day")),
        (cftime_datetime(2023, 12, 1, calendar="360_day"), cftime_datetime(2023, 12, 2, calendar="360_day")),
        (datetime(2023, 12, 1), datetime(2023, 12, 2)),
        (np.datetime64(datetime(2023, 12, 1)), np.datetime64(datetime(2023, 12, 2))),
    ],
)
def test_time_interval_initialization(left, right):
    """Test that TimeInterval can be initialized with valid inputs."""
    interval = TimeInterval(left, right)
    assert interval.left == left
    assert interval.right == right

    with pytest.raises(ValueError):
        TimeInterval(right, left)


@given(pst.time.time_interval())
def test_time_interval_contains(interval):
    left = 0
    right = timedelta_to_float(interval.right - interval.left)
    middle = timedelta_to_float(interval.right - interval.left) / 2

    assert interval.is_all_time_in_interval(left)
    assert interval.is_all_time_in_interval(right)
    assert interval.is_all_time_in_interval(middle)


@given(pst.time.time_interval(calendar="365_day"), pst.time.time_interval(calendar="365_day"))
def test_time_interval_intersection_commutative(interval1, interval2):
    assert interval1.intersection(interval2) == interval2.intersection(interval1)


@given(pst.time.time_interval())
def test_time_interval_intersection_with_self(interval):
    assert interval.intersection(interval) == interval


def test_time_interval_repr():
    """Test the string representation of TimeInterval."""
    interval = TimeInterval(datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 2, 12, 0))
    expected = "TimeInterval(left=datetime.datetime(2023, 1, 1, 12, 0), right=datetime.datetime(2023, 1, 2, 12, 0))"
    assert repr(interval) == expected


@given(pst.time.time_interval())
def test_time_interval_equality(interval):
    assert interval == interval


@pytest.mark.parametrize(
    "interval1,interval2,expected",
    [
        pytest.param(
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 2, calendar="gregorian"), cftime_datetime(2023, 1, 4, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 2, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            id="overlapping intervals",
        ),
        pytest.param(
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 5, calendar="gregorian"), cftime_datetime(2023, 1, 6, calendar="gregorian")
            ),
            None,
            id="non-overlapping intervals",
        ),
        pytest.param(
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 2, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 2, calendar="gregorian")
            ),
            id="intervals with same start time",
        ),
        pytest.param(
            TimeInterval(
                cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 2, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            TimeInterval(
                cftime_datetime(2023, 1, 2, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
            ),
            id="intervals with same end time",
        ),
    ],
)
def test_time_interval_intersection(interval1, interval2, expected):
    """Test the intersection of two time intervals."""
    result = interval1.intersection(interval2)
    if expected is None:
        assert result is None
    else:
        assert result.left == expected.left
        assert result.right == expected.right


def test_time_interval_intersection_different_calendars():
    interval1 = TimeInterval(
        cftime_datetime(2023, 1, 1, calendar="gregorian"), cftime_datetime(2023, 1, 3, calendar="gregorian")
    )
    interval2 = TimeInterval(
        cftime_datetime(2023, 1, 1, calendar="365_day"), cftime_datetime(2023, 1, 3, calendar="365_day")
    )
    with pytest.raises(ValueError, match="TimeIntervals are not compatible."):
        interval1.intersection(interval2)


@pytest.mark.parametrize(
    "td,expected",
    [
        pytest.param(np.timedelta64(1, "s"), np.timedelta64(1, "s"), id="noop"),
        pytest.param(timedelta(days=5), np.timedelta64(5, "D"), id="single unit"),
        pytest.param(timedelta(days=5, seconds=30), np.timedelta64(5, "D") + np.timedelta64(30, "s"), id="mixed units"),
        pytest.param(timedelta(days=0), np.timedelta64(0, "s"), id="zero timedelta"),
        pytest.param(
            timedelta(seconds=-2), np.timedelta64(-2, "s"), id="negative timedelta"
        ),  # included because timedelta(seconds=-2) -> timedelta(days=-1, seconds=86398)
    ],
)
def test_maybe_convert_python_timedelta_to_numpy(td, expected):
    result = maybe_convert_python_timedelta_to_numpy(td)
    assert result == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (timedelta(days=1), 24 * 60 * 60),
        (np.timedelta64(1, "D"), 24 * 60 * 60),
        (3600.0, 3600.0),
    ],
)
def test_timedelta_to_float(input, expected):
    assert timedelta_to_float(input) == expected


def test_timedelta_to_float_exceptions():
    with pytest.raises((ValueError, TypeError)):
        timedelta_to_float("invalid_type")


@given(pst.time.datetime_various())
def test_datetime_get_cf_attrs(dt):
    attrs = _get_cf_attrs(dt)
    assert "seconds" in attrs["units"]


@pytest.mark.parametrize(
    "t1,t2,expected",
    [
        pytest.param(datetime(2023, 1, 1), datetime(2023, 1, 2), True, id="datetime"),
        pytest.param(np.timedelta64(1, "s"), np.timedelta64(2, "s"), True, id="timedelta64"),
        pytest.param(np.timedelta64(1, "s"), datetime(2023, 1, 2), False, id="timedelta64 vs datetime"),
        pytest.param(
            datetime(2023, 1, 1), cftime_datetime(2023, 1, 2, calendar="noleap"), False, id="datetime vs cftime"
        ),
        pytest.param(
            cftime_datetime(2023, 1, 1, calendar="noleap"),
            cftime_datetime(2023, 1, 2, calendar="360_day"),
            False,
            id="cftime different calendars",
        ),
    ],
)
def test_is_compatible(t1, t2, expected):
    assert is_compatible(t1, t2) is expected


def test_is_compatible_propagates_unexpected_errors():
    class Exploding:
        def __sub__(self, other):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        is_compatible(Exploding(), Exploding())


def test_maybe_convert_python_timedelta_to_numpy_invalid():
    with pytest.raises(ValueError, match="Could not convert"):
        maybe_convert_python_timedelta_to_numpy("not a timedelta")


def test_timedelta_to_float_object_array_invalid():
    with pytest.raises(ValueError, match="Expected a timedelta-like object"):
        timedelta_to_float(np.array([timedelta(seconds=1), "not a timedelta"], dtype=object))
