from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

import cftime
import numpy as np
import numpy.typing as npt

from parcels._repr_utils import timeinterval_repr

if TYPE_CHECKING:
    from parcels._typing import TimeLike

T = TypeVar("T", bound="TimeLike")


class TimeInterval:
    """A class representing a time interval between two datetime or np.timedelta64 objects.

    Parameters
    ----------
    left : np.datetime64 or cftime.datetime or np.timedelta64
        The left endpoint of the interval.
    right : np.datetime64 or cftime.datetime or np.timedelta64
        The right endpoint of the interval.

    Notes
    -----
    For the purposes of this codebase, the interval can be thought of as closed on the left and right.
    """

    def __init__(self, left: T, right: T) -> None:
        if not isinstance(left, (np.timedelta64, datetime, cftime.datetime, np.datetime64)):
            raise ValueError(
                f"Expected right to be a np.timedelta64, datetime, cftime.datetime, or np.datetime64. Got {type(left)}."
            )
        if not isinstance(right, (np.timedelta64, datetime, cftime.datetime, np.datetime64)):
            raise ValueError(
                f"Expected right to be a np.timedelta64, datetime, cftime.datetime, or np.datetime64. Got {type(right)}."
            )
        if left >= right:
            raise ValueError(f"Expected left to be strictly less than right, got left={left} and right={right}.")

        if not is_compatible(left, right):
            raise ValueError(f"Expected left and right to be compatible, got left={left} and right={right}.")

        self.left = left
        self.right = right

    @property
    def time_length_as_flt(self):
        if isinstance(self.right - self.left, np.timedelta64):
            return timedelta_to_float(self.right - self.left)
        if isinstance(self.right - self.left, timedelta):
            return (self.right - self.left).total_seconds()
        return self.right - self.left

    def __contains__(self, item: T) -> bool:
        return self.left <= item <= self.right

    def is_all_time_in_interval(self, time: float | np.ndarray):
        item = np.atleast_1d(time)
        return (0 <= item).all() and (item <= self.time_length_as_flt).all()

    def __repr__(self) -> str:
        return timeinterval_repr(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeInterval):
            return False
        return self.left == other.left and self.right == other.right

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def intersection(self, other: TimeInterval) -> TimeInterval | None:
        """Return the intersection of two time intervals. Returns None if there is no overlap."""
        if not is_compatible(self.left, other.left):
            raise ValueError("TimeIntervals are not compatible.")
        if not is_compatible(self.right, other.right):
            raise ValueError("TimeIntervals are not compatible.")

        start = max(self.left, other.left)
        end = min(self.right, other.right)

        return TimeInterval(start, end) if start <= end else None

    def get_cf_attrs(self) -> dict[Literal["units", "calendar"], str]:
        """Return the cf-attrs that would correspond to x seconds from the left edge."""
        return _get_cf_attrs(self.left)


def _get_cf_attrs(dt: TimeLike) -> dict[Literal["units", "calendar"], str]:
    if isinstance(dt, cftime.datetime):
        dt = cast(cftime.datetime, dt)
        return {"units": f"seconds since {dt.strftime(dt.format)}", "calendar": dt.calendar}

    if isinstance(dt, np.timedelta64):
        return {"units": "seconds"}

    from pandas import Timestamp

    if isinstance(dt, np.datetime64):
        dt = Timestamp(dt)

    if isinstance(dt, (Timestamp, datetime)):
        dt_cf = cftime.datetime(
            year=dt.year,
            month=dt.month,
            day=dt.day,
            hour=dt.hour,
            minute=dt.minute,
            second=dt.second,
            microsecond=dt.microsecond,
            calendar="gregorian",  # What is the cftime proleptic_gregorian calendar? is that relevant here?
        )
        return _get_cf_attrs(dt_cf)

    raise NotImplementedError(f"Not implemented for time object {type(dt)=!r}")


def is_compatible(
    t1: datetime | cftime.datetime | np.timedelta64, t2: datetime | cftime.datetime | np.timedelta64
) -> bool:
    """
    Defines whether two datetime or np.timedelta64 objects are compatible in the context
    of being left and right sides of an interval.
    """
    # Ensure if either is a timedelta64, both must be
    if isinstance(t1, np.timedelta64) ^ isinstance(t2, np.timedelta64):
        return False

    try:
        t1 - t2  # type: ignore[operator]  # probing compatibility at runtime is the point
    except Exception:
        return False
    else:
        return True


def get_datetime_type_calendar(
    example_datetime: TimeLike,
) -> tuple[type, str | None]:
    """Get the type and calendar of a datetime object.

    Parameters
    ----------
    example_datetime : datetime, cftime.datetime, or np.datetime64
        The datetime object to check.

    Returns
    -------
    tuple[type, str | None]
        A tuple containing the type of the datetime object and its calendar.
        The calendar will be None if the datetime object is not a cftime datetime object.
    """
    calendar = None
    try:
        calendar = example_datetime.calendar
    except AttributeError:
        # datetime isn't a cftime datetime object
        pass
    return type(example_datetime), calendar


_TD_PRECISION_GETTER_FOR_UNIT: tuple[tuple[Callable[[timedelta], int], Literal["D", "s", "us"]], ...] = (
    (lambda dt: dt.days, "D"),
    (lambda dt: dt.seconds, "s"),
    (lambda dt: dt.microseconds, "us"),
)


def maybe_convert_python_timedelta_to_numpy(dt: timedelta | np.timedelta64) -> np.timedelta64:
    if isinstance(dt, np.timedelta64):
        return dt

    try:
        dts = []
        for get_value_for_unit, np_unit in _TD_PRECISION_GETTER_FOR_UNIT:
            value = get_value_for_unit(dt)
            if value != 0:
                dts.append(np.timedelta64(value, np_unit))

        if dts:
            return np.sum(dts)
        else:
            return np.timedelta64(0, "s")
    except Exception as e:
        raise ValueError(f"Could not convert {dt!r} to np.timedelta64.") from e


@overload
def timedelta_to_float(dt: float | timedelta | np.timedelta64) -> float: ...
@overload
def timedelta_to_float(dt: npt.NDArray[Any]) -> npt.NDArray[np.float64]: ...
def timedelta_to_float(
    dt: float | timedelta | np.timedelta64 | npt.NDArray[Any],
) -> float | npt.NDArray[np.float64]:
    """Convert a timedelta to a float in seconds."""
    if isinstance(dt, timedelta):
        return dt.total_seconds()
    if isinstance(dt, np.timedelta64):
        return float(dt / np.timedelta64(1, "s"))
    if hasattr(dt, "dtype"):
        if np.issubdtype(dt.dtype, np.timedelta64):  # in case of array
            return (cast("npt.NDArray[Any]", dt) / np.timedelta64(1, "s")).astype(float)
        elif np.issubdtype(dt.dtype, np.object_):  # in case of array of timedeltas
            try:
                f = np.vectorize(lambda x: x.total_seconds())
                return f(dt)
            except Exception as e:
                raise ValueError(f"Expected a timedelta-like object, got {dt!r}.") from e
    return float(dt)


def float_to_datelike(dt: float, time_interval) -> np.datetime64 | np.timedelta64 | cftime.datetime:
    """Convert a float time (in seconds from the start of the time_interval) to a datetime (if time_interval is a datetime object) or timedelta (otherwise)"""
    if time_interval:
        result = np.timedelta64(int(dt), "s") + time_interval.left
        if isinstance(result, cftime.datetime):
            return result
        return result.astype("datetime64[s]")
    else:
        return np.timedelta64(int(dt), "s")
