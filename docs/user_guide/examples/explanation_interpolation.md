# 📖 Interpolators Overview and API

Interpolation is an important functionality of Parcels. On this page we will discuss the way it is
implemented in Parcels and how to write a custom interpolator function.

When we want to know the state of particles in an environmental field, such as temperature or velocity,
we _evaluate_ the {py:obj}`parcels.Field` at the particles real position in time and space (`t`, `z`, `y`, `x`).
In Parcels we can do this using square brackets:

```
particles.temperature = fieldset.temperature[particles]
```

````{note}
The statement above is shorthand for
```python
particles.temperature = fieldset.temperature[
    particles.t, particles.z, particles.y, particles.x, particles
]
```
where the `particles` argument at the end provides the grid search algorithm with a first guess for the element indices to interpolate on.

If you want to sample at a different location, or time, that is not necessarily close to the particles location, you can use
```python
particles.temperature = fieldset.temperature[t, z, y, x]
```
but this could be slower for curvilinear and unstructured Grids because the entire grid needs to be searched.
````

The values of the `temperature` field at the particles' positions are determined using an interpolation
method. This interpolation method defines how the discretized values of the `parcels.Field` should
relate to the value at any point within a grid cell.

Each `parcels.Field` is defined on a (structured) `parcels.XGrid` or (unstructured) `parcels.UXGrid`.
The interpolation function takes information about the particles position relative to this grid (`grid_positions`),
as well as the values of the grid points of the `parcels.Field` in time and space, to calculate
the requested value at the particles location.

## Interpolator API

The interpolators included in Parcels are designed for common interpolation schemes in Parcels simulations; see the [Using the built-in interpolators tutorial](./tutorial_interpolation.ipynb).

If we want to create a custom interpolation method, we need to look at the interpolator API. Each interpolator is a class that inherits from either the `ScalarInterpolator` or `VectorInterpolator` class. The `ScalarInterpolator` class is used for scalar fields, such as temperature or salinity, while the `VectorInterpolator` class is used for vector fields, such as velocity.

An interpolator class must have a `.interp()` method with the following signature:

```python
def interp(
    self,
    particle_positions: dict[str, float | np.ndarray],
    grid_positions: dict[ptyping.XgridAxis, dict[str, int | float | np.ndarray]],
    field: Field,
): ...
```

The `particle_positions` dictionary contains:

```python
particle_positions = {"t": t, "z": z, "y": y, "x": x}
```

For structured (`X`) grids, the `grid_positions` dictionary contains:

```python
grid_positions = {
    "T": {"index": ti, "bcoord": tau},
    "Z": {"index": zi, "bcoord": zeta},
    "Y": {"index": yi, "bcoord": eta},
    "X": {"index": xi, "bcoord": xsi},
}
```

where `index` is the grid index in the corresponding dimension, and `bcoord` is the barycentric coordinate in the grid cell.

For unstructured (`UX`) grids, the same dictionary is defined as:

```python
grid_positions = {
    "T": {"index": ti, "bcoord": tau},
    "Z": {"index": zi, "bcoord": zeta},
    "FACE": {"index": fi, "bcoord": bcoord},
}
```

The `.interp()` method should return a float (in the case of a `ScalarInterpolator` or a tuple of three floats `(u, v, w)` in the case of a `VectorInterpolator`).

Writing custom interpolators is not trivial, so we recommend that you have a look at the built-in [structured-grid interpolators](https://github.com/Parcels-code/Parcels/blob/main/src/parcels/interpolators/_xinterpolators.py) or the [unstructured-grid interpolators](https://github.com/Parcels-code/Parcels/blob/main/src/parcels/interpolators/_uxinterpolators.py) to see how they are implemented.
