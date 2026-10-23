from __future__ import annotations

import numpy as np
import pytest
from re_assert import Matches

from parcels import Field, VectorField
from parcels._core.fieldset import FieldSet
from parcels._core.model import StructuredModelData
from parcels._core.warnings import FieldEvalWarning
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels._datasets.structured.generic import T as T_structured
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels._datasets.unstructured.generic import _ux_constant_flow_face_centered_2D
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels._python import NOTSET
from parcels.interpolators import (
    UxConstantFaceConstantZC,
)


def test_field_init_param_types():
    data = datasets_structured["ds_2d_left"]
    model = StructuredModelData.from_sgrid_conventions(data, vector_fields=NOTSET)

    with pytest.raises(TypeError, match="Expected a string for variable name, got int instead."):
        Field(name=123, model=model)

    for name in ["a b", "123"]:
        with pytest.raises(
            ValueError,
            match=r"Received invalid Python variable name.*: not a valid identifier. HINT: avoid using spaces, special characters, and starting with a number.",
        ):
            Field(name=name, model=model)

    with pytest.raises(
        ValueError,
        match=r"Received invalid Python variable name.*: it is a reserved keyword. HINT: avoid using the following names:.*",
    ):
        Field(name="while", model=model)


def test_field_repr(fieldset):
    Matches(r"Field\(name=.*, model=.*\)").assert_matches(repr(fieldset.U))


def test_vectorfield_repr(fieldset):
    Matches(r"\<.*VectorField object at.*\>").assert_matches(repr(fieldset.UV))


# TODO: Move to test_model.py ?
def test_field_init_fail_on_float_time_dim():
    """Test that accessing time_interval fails when dataset has float time dimension.

    (users are expected to use timedelta64 or datetime).
    Time validation has moved from Field.__init__ to ModelData.time_interval.
    """
    ds = datasets_structured["ds_2d_left"].copy()
    ds["time"] = (
        ds["time"].dims,
        np.arange(0, T_structured, dtype="float64"),
        ds["time"].attrs,
    )

    model = StructuredModelData.from_sgrid_conventions(ds, vector_fields=NOTSET)
    with pytest.raises(
        ValueError,
        match=r"Are you sure that the time dimension on the xarray dataset is stored as timedelta, datetime or cftime datetime objects\?",
    ):
        _ = model.time_interval


# TODO: Move to test_model.py as test_model_time_interval() ?
def test_field_time_interval():
    """Test that field.time_interval delegates correctly to model.time_interval."""
    data = datasets_structured["ds_2d_left"]
    model = StructuredModelData.from_sgrid_conventions(data, vector_fields=NOTSET)
    field = Field(name="data_g", model=model)
    assert field.time_interval.left == np.datetime64("2000-01-01")
    assert field.time_interval.right == np.datetime64("2001-01-01")


def test_vectorfield_init_different_time_intervals():
    # Tests that a VectorField raises a ValueError if the component fields have different time domains.
    ...


def test_field_invalid_interpolator():
    ds = datasets_structured["ds_2d_left"]
    model = StructuredModelData.from_sgrid_conventions(ds, vector_fields=NOTSET)
    field = Field(name="data_g", model=model)

    def not_a_scalar_interpolator(particle_positions, grid_positions, field):
        return 0.0

    # Interpolators must now be ScalarInterpolator instances, not plain callables
    with pytest.raises(ValueError, match="interp_method must be a `ScalarInterpolator` object"):
        field.interp_method = not_a_scalar_interpolator


def test_vectorfield_invalid_interpolator():
    ds = datasets_structured["ds_2d_left"]
    model = StructuredModelData.from_sgrid_conventions(ds, vector_fields=NOTSET)
    fields = {f.name: f for f in model.construct_fields()}
    U = fields["U_A_grid"]
    V = fields["V_A_grid"]

    def not_a_vector_interpolator(particle_positions, grid_positions, field):
        return 0.0

    # VectorField interp_method must be a VectorInterpolator instance, not a plain callable
    with pytest.raises(ValueError, match="interp_method must be a `VectorInterpolator` object"):
        VectorField(
            name="UV",
            U=U,
            V=V,
            interp_method=not_a_vector_interpolator,
        )


def test_field_unstructured_z_linear():
    """Tests correctness of piecewise constant and piecewise linear interpolation methods on an unstructured grid with a vertical coordinate.
    The example dataset is a FESOM2 square Delaunay grid with uniform z-coordinate. Cell centered and layer registered data are defined to be
    linear functions of the vertical coordinate. This allows for testing of exactness of the interpolation methods.
    """
    ds = datasets_unstructured["fesom2_square_delaunay_uniform_z_coordinate"].copy(deep=True)
    ds = ds.rename(
        {
            "nz": "zf",  # Vertical Interface
            "nz1": "zc",  # Vertical Center
        }
    ).set_index(zf="zf", zc="zc")

    # Change the pressure values to be linearly dependent on the vertical coordinate
    for k, z in enumerate(ds.coords["zc"]):
        ds["p"].values[:, k, :] = z

    # Change the vertical velocity values to be linearly dependent on the vertical coordinate
    for k, z in enumerate(ds.coords["zf"]):
        ds["W"].values[:, k, :] = z

    fieldset = FieldSet.from_ugrid_conventions(ds)
    # Note that the vertical coordinate is required to be the position of the layer interfaces ("nz"), not the mid-layers ("nz1")
    P = fieldset.p
    W = fieldset.W

    # Test above first cell center - for piecewise constant, should return the depth of the first cell center
    assert np.isclose(
        P.eval(t=[0], z=[10.0], y=[30.0], x=[30.0]),
        55.555557,
    )
    # Test below first cell center, but in the first layer  - for piecewise constant, should return the depth of the first cell center
    assert np.isclose(
        P.eval(t=[0], z=[65.0], y=[30.0], x=[30.0]),
        55.555557,
    )
    # Test bottom layer  - for piecewise constant, should return the depth of the of the bottom layer cell center
    assert np.isclose(
        P.eval(t=[0], z=[900.0], y=[30.0], x=[30.0]),
        944.44445801,
    )

    assert np.isclose(
        W.eval(t=[0], z=[10.0], y=[30.0], x=[30.0]),
        10.0,
    )
    assert np.isclose(
        W.eval(t=[0], z=[65.0], y=[30.0], x=[30.0]),
        65.0,
    )
    assert np.isclose(
        W.eval(t=[0], z=[900.0], y=[30.0], x=[30.0]),
        900.0,
    )


def test_field_constant_in_time():
    """Tests field evaluation for a field with no time interval (i.e., constant in time)."""
    fieldset = FieldSet.from_ugrid_conventions(datasets_unstructured["stommel_gyre_delaunay"])
    # Note that the vertical coordinate is required to be the position of the layer interfaces ("nz"), not the mid-layers ("nz1")
    P = fieldset.p
    assert isinstance(P.interp_method, UxConstantFaceConstantZC)

    # Assert that the field can be evaluated at any time, and returns the same value
    time = np.datetime64("2000-01-01T00:00:00")
    P1 = P.eval(t=time, z=[10.0], y=[30.0], x=[30.0])
    P2 = P.eval(
        t=time + np.timedelta64(1, "D"),
        z=[10.0],
        y=[30.0],
        x=[30.0],
    )
    assert np.isclose(P1, P2)


@pytest.mark.parametrize(
    "field_name, location, expected",
    [
        ("U", (0.0, 0.0, 0.0, 5e6), 0.0),
        ("UV", (0.0, 0.0, 0.0, 5e6), [[0.0], [0.0]]),
        ("U", (0.0, 0.0, 5e6, 0.0), 0.0),
        ("UV", (0.0, 0.0, 5e6, 0.0), [[0.0], [0.0]]),
        ("U", (0.0, 5e6, 0.0, 0.0), 0.0),
        ("UV", (0.0, 5e6, 0.0, 0.0), [[0.0], [0.0]]),
    ],
)
def test_field_eval_out_of_bounds_structured(field_name, location, expected):
    """Test that Field.eval returns IndexError when queried outside the grid boundaries."""
    # eval outside of bounds should return 0.0
    ds = simple_UV_dataset(mesh_str="flat")
    fieldset = FieldSet.from_sgrid_conventions(ds)
    fieldset.U.data[:] = 1.0
    fieldset.V.data[:] = 2.0
    field = getattr(fieldset, field_name)
    with pytest.warns(
        FieldEvalWarning,
        match="Some interpolated values are out-of-bounds. These values are set to 0. Treat carefully.",
    ):
        np.testing.assert_allclose(field.eval(*location), expected)


@pytest.mark.parametrize(
    "field_name, location, expected",
    [
        ("U", (0.0, 0.0, 0.0, 5e6), 0.0),
        ("UV", (0.0, 0.0, 0.0, 5e6), [[0.0], [0.0]]),
        ("U", (0.0, 0.0, 5e6, 0.0), 0.0),
        ("UV", (0.0, 0.0, 5e6, 0.0), [[0.0], [0.0]]),
        ("U", (0.0, 5e6, 0.0, 0.0), 0.0),
        ("UV", (0.0, 5e6, 0.0, 0.0), [[0.0], [0.0]]),
    ],
)
def test_field_eval_out_of_bounds_unstructured(field_name, location, expected):
    """Test that Field.eval returns IndexError when queried outside the grid boundaries."""
    ds = _ux_constant_flow_face_centered_2D()
    fieldset = FieldSet.from_ugrid_conventions(ds)
    fieldset.U.data[:] = 1.0
    fieldset.V.data[:] = 2.0
    field = getattr(fieldset, field_name)
    with pytest.warns(
        FieldEvalWarning,
        match="Some interpolated values are out-of-bounds. These values are set to 0. Treat carefully.",
    ):
        np.testing.assert_allclose(field.eval(*location), expected)


def test_field_unstructured_grid_creation(): ...


def test_field_interpolation(): ...


def test_field_interpolation_out_of_spatial_bounds(): ...


def test_field_interpolation_out_of_time_bounds(): ...
