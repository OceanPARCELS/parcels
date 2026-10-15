# isort: skip_file
from importlib.metadata import version as _version

try:
    __version__ = _version("parcels")
except Exception:
    # Local copy or not installed with setuptools.
    __version__ = "unknown"

from parcels._core.fieldset import FieldSet
from parcels._core.particleset import ParticleSet
from parcels._core.particlefile import ParticleFile, read_particlefile
from parcels._compat_v3 import particlefile_to_v3_zarr
from parcels._core.particle import (
    Variable,
    Particle,
    ParticleClass,
)
from parcels._core.field import Field, VectorField
from parcels._core.basegrid import BaseGrid
from parcels._core.uxgrid import UxGrid
from parcels._core.xgrid import XGrid
from parcels._core.mesh import SphericalMesh

from parcels._core.statuscodes import (
    AllParcelsErrorCodes,
    FieldInterpolationError,
    FieldOutOfBoundError,
    FieldSamplingError,
    KernelError,
    OutsideTimeInterval,
    StatusCode,
)
from parcels._core.warnings import (
    FieldSetWarning,
    FileWarning,
    KernelWarning,
    ParticleSetWarning,
    FieldEvalWarning,
)
from parcels._core.utils.kernel_linting import validate_kernel, KernelValidationError
from parcels._logger import logger
from . import convert
from . import kernels

__all__ = [  # noqa: RUF022
    # Core classes
    "FieldSet",
    "ParticleSet",
    "ParticleFile",
    "Variable",
    "Particle",
    "ParticleClass",
    "Field",
    "VectorField",
    "BaseGrid",
    "UxGrid",
    "XGrid",
    "SphericalMesh",
    # Status codes and errors
    "AllParcelsErrorCodes",
    "FieldInterpolationError",
    "FieldOutOfBoundError",
    "FieldSamplingError",
    "KernelError",
    "OutsideTimeInterval",
    "StatusCode",
    # Warnings
    "FieldEvalWarning",
    "FieldSetWarning",
    "FileWarning",
    "KernelWarning",
    "ParticleSetWarning",
    # Utilities
    "logger",
    "read_particlefile",
    "particlefile_to_v3_zarr",
    "convert",
    # kernels
    "kernels",
    # Kernel validation
    "validate_kernel",
    "KernelValidationError",
]
