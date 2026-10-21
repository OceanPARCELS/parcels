from __future__ import annotations

import operator
from typing import Any

import numpy as np

from parcels._compat import _attrgetter_helper
from parcels._core.statuscodes import StatusCode
from parcels._core.utils.string import _assert_str_and_python_varname
from parcels._repr_utils import particleclass_repr, variable_repr

__all__ = ["Particle", "ParticleClass", "Variable"]
_TO_WRITE_OPTIONS = [True, False]


class Variable:
    """Descriptor class that delegates data access to particle data.

    Parameters
    ----------
    name : str
        Variable name as used within kernels
    dtype :
        Data type (numpy.dtype) of the variable
    initial :
        Initial value of the variable. Note that this can also be a Field object,
        which will then be sampled at the location of the particle
    to_write : bool, optional
        Controls whether Variable is written to output file.
    attrs : dict, optional
        Attributes to be stored with the variable when written to file. This can include metadata such as units, long_name, etc.
    """

    def __init__(
        self,
        name,
        dtype: np.dtype[Any] | type[np.generic] = np.float32,
        initial=0,
        to_write: bool = True,
        attrs: dict | None = None,
    ):
        _assert_str_and_python_varname(name)

        try:
            dtype = np.dtype(dtype)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Variable dtype must be a valid numpy dtype. Got {dtype=!r}") from e

        if to_write not in _TO_WRITE_OPTIONS:
            raise ValueError(f"to_write must be one of {_TO_WRITE_OPTIONS!r}. Got {to_write=!r}")

        if attrs is None:
            attrs = {}

        if not to_write:
            if attrs != {}:
                raise ValueError(f"Attributes cannot be set if {to_write=!r}.")

        self._name = name
        self.dtype = dtype
        self.initial = initial
        self.to_write = to_write
        self.attrs = attrs

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return variable_repr(self)


class ParticleClass:
    """Define a class of particles. This is used to generate the particle data which is then used in the simulation.

    Parameters
    ----------
    variables : list[Variable]
        List of Variable objects that define the particle's attributes.

    """

    def __init__(self, variables: list[Variable]):
        if not isinstance(variables, list):
            raise TypeError(f"Expected list of Variable objects, got {type(variables)}")
        if not all(isinstance(var, Variable) for var in variables):
            raise ValueError(f"All items in variables must be instances of Variable. Got {variables=!r}")

        self.variables = variables

    def __repr__(self):
        return particleclass_repr(self)

    def add_variable(self, variable: Variable | list[Variable]):
        """Add a new variable to the Particle class. This returns a new Particle class with the added variable(s).

        Parameters
        ----------
        variable : Variable or list[Variable]
            Variable or list of Variables to be added to the Particle class.
            If a list is provided, all variables will be added to the class.
        """
        if isinstance(variable, Variable):
            variable = [variable]

        for var in variable:
            if not isinstance(var, Variable):
                raise TypeError(f"Expected Variable, got {type(var)}")

        _assert_no_duplicate_variable_names(existing_vars=self.variables, new_vars=variable)

        return ParticleClass(variables=self.variables + variable)


def _assert_no_duplicate_variable_names(*, existing_vars: list[Variable], new_vars: list[Variable]):
    existing_names = {var.name for var in existing_vars}
    for var in new_vars:
        if var.name in existing_names:
            raise ValueError(f"Variable name already exists: {var.name}")


def get_default_particle(spatial_dtype: type[np.float32] | type[np.float64]) -> ParticleClass:
    if spatial_dtype not in [np.float32, np.float64]:
        raise ValueError(f"spatial_dtype must be np.float32 or np.float64. Got {spatial_dtype=!r}")

    return ParticleClass(
        variables=[
            Variable(
                "t",
                dtype=np.float64,
                attrs={
                    "standard_name": "time",
                    "units": "seconds",
                    "axis": "T",
                },  # "units" and "calendar" gets updated/set if working with cftime time domain
            ),
            Variable(
                "z",
                dtype=spatial_dtype,
                attrs={"standard_name": "vertical coordinate", "units": "m", "positive": "down"},
            ),
            Variable(
                "y",
                dtype=spatial_dtype,
                attrs={
                    "standard_name": "latitude",
                    "units": "degrees_north",
                    "axis": "Y",
                },  # TODO v4: https://github.com/Parcels-code/Parcels/issues/2720
            ),
            Variable(
                "x",
                dtype=spatial_dtype,
                attrs={
                    "standard_name": "longitude",
                    "units": "degrees_east",
                    "axis": "X",
                },  # TODO v4: https://github.com/Parcels-code/Parcels/issues/2720
            ),
            Variable("dz", dtype=spatial_dtype, to_write=False),
            Variable("dy", dtype=spatial_dtype, to_write=False),
            Variable("dx", dtype=spatial_dtype, to_write=False),
            Variable(
                "particle_id",
                dtype=np.int64,
                attrs={
                    "long_name": "Unique identifier for each particle",
                    "cf_role": "trajectory_id",
                },
            ),
            Variable("dt", dtype=np.float64, initial=1.0, to_write=False),
            Variable("state", dtype=np.int32, initial=StatusCode.Success, to_write=False),
        ]
    )


Particle = get_default_particle(np.float32)
"""The default Particle used in Parcels simulations."""


def create_particle_data(
    *,
    pclass: ParticleClass,
    nparticles: int,
    ngrids: int,
    initial: dict[str, np.ndarray] | None = None,
):
    if initial is None:
        initial = {}

    variables = {var.name: var for var in pclass.variables}

    assert "ei" not in initial, "'ei' is for internal use, and is unique since is only non 1D array"

    dtypes = {var.name: var.dtype for var in variables.values()}

    for var_name in initial:
        if var_name not in variables:
            raise ValueError(f"Variable {var_name} is not defined in the ParticleClass.")

        values = initial[var_name]
        if values.shape != (nparticles,):
            raise ValueError(f"Initial value for {var_name} must have shape ({nparticles},). Got {values.shape=}")

        initial[var_name] = values.astype(dtypes[var_name])

    data = {"ei": np.zeros((nparticles, ngrids), dtype=np.int32), **initial}

    vars_to_create = {k: v for k, v in variables.items() if k not in data}

    for var in vars_to_create.values():
        if isinstance(var.initial, operator.attrgetter):
            name_to_copy = var.initial(_attrgetter_helper)
            data[var.name] = data[name_to_copy].copy()
        else:
            data[var.name] = np.full(
                shape=(nparticles,),
                fill_value=var.initial,
                dtype=var.dtype,
            )
    return data
