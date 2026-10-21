"""Handling of Errors and particle status codes"""

__all__ = [
    "AllParcelsErrorCodes",
    "FieldOutOfBoundError",
    "FieldSamplingError",
    "KernelError",
    "OutsideTimeInterval",
    "StatusCode",
    "_raise_field_interpolation_error",
    "_raise_field_out_of_bound_error",
    "_raise_field_out_of_bound_surface_error",
    "_raise_general_error",
    "_raise_grid_searching_error",
    "_raise_outside_time_interval_error",
]


class StatusCode:
    """Class defining the status codes for particles.state."""

    Success = 0
    EndofLoop = 1
    Repeat = 20
    Delete = 30
    StopExecution = 40
    StopAllExecution = 41
    Error = 50
    ErrorInterpolation = 51
    ErrorGridSearching = 52
    ErrorOutOfBounds = 60
    ErrorThroughSurface = 61
    ErrorOutsideTimeInterval = 70


class FieldInterpolationError(RuntimeError):
    """Utility error class to propagate NaN field interpolation."""

    pass


def _raise_field_interpolation_error(z, y, x):
    raise FieldInterpolationError(f"Field interpolation returned NaN at (z={z}, y={y}, x={x})")


class FieldOutOfBoundError(RuntimeError):
    """Utility error class to propagate out-of-bound field sampling."""

    pass


def _raise_field_out_of_bound_error(z, y, x):
    raise FieldOutOfBoundError(f"Field sampled out-of-bound, at (z={z}, y={y}, x={x})")


class FieldOutOfBoundSurfaceError(RuntimeError):
    """Utility error class to propagate out-of-bound field sampling at the surface."""

    pass


def _raise_field_out_of_bound_surface_error(z: float | None, y: float | None, x: float | None) -> None:
    def format_out(val):
        return "unknown" if val is None else val

    raise FieldOutOfBoundSurfaceError(
        f"Field sampled out-of-bound at the surface, at (z={format_out(z)}, y={format_out(y)}, x={format_out(x)})"
    )


class FieldSamplingError(RuntimeError):
    """Utility error class to propagate field sampling errors."""

    pass


class GridSearchingError(RuntimeError):
    """Utility error class to propagate grid searching errors."""

    pass


def _raise_grid_searching_error(z, y, x):
    raise GridSearchingError(f"Grid searching failed at (z={z}, y={y}, x={x})")


class GeneralError(RuntimeError):
    """Utility error class to propagate general errors."""

    pass


def _raise_general_error(z, y, x):
    raise GeneralError(f"General error occurred at (z={z}, y={y}, x={x})")


class OutsideTimeInterval(RuntimeError):
    """Utility error class to propagate erroneous time extrapolation sampling."""

    def __init__(self, time, field=None):
        message = f"{field.name if field else 'Field'} sampled outside time domain at time {time}."
        super().__init__(message)


def _raise_outside_time_interval_error(time: float, field=None):
    raise OutsideTimeInterval(time, field)


class KernelError(RuntimeError):
    """General particles kernel error with optional custom message."""

    def __init__(self, particles, fieldset=None, msg=None):
        message = f"{particles.state}\nParticle {particles}\nTime: {particles.time}\ntimestep dt: {particles.dt}\n"
        if msg:
            message += msg
        super().__init__(message)


AllParcelsErrorCodes: dict[type[Exception], int] = {
    FieldInterpolationError: StatusCode.ErrorInterpolation,
    FieldOutOfBoundError: StatusCode.ErrorOutOfBounds,
    FieldOutOfBoundSurfaceError: StatusCode.ErrorThroughSurface,
    GridSearchingError: StatusCode.ErrorGridSearching,
    OutsideTimeInterval: StatusCode.ErrorOutsideTimeInterval,
    KernelError: StatusCode.Error,
    GeneralError: StatusCode.Error,
}
