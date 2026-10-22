---
file_format: mystnb
kernelspec:
  name: python3
---

# 📖 Writing your own convert function

Parcels reads structured-grid model data through {py:func}`parcels.FieldSet.from_sgrid_conventions`. This function
does not guess how your model grid is laid out. Instead, it reads [SGRID](https://sgrid.github.io/sgrid/) metadata
attached to the dataset, which provides information on which dimensions hold the cell corners, which dimensions hold the cell centres, and how
they line up. See the [Grids explanation](./explanation_grids.ipynb#nodes-edges-and-faces) for more on where data can
sit on a grid cell.

Most model output does not (yet) come with SGrid-compliant metadata. For that reason, Parcels ships a set of built-in `convert` functions in
{py:mod}`parcels.convert` (for example {py:func}`parcels.convert.nemo_to_sgrid`,
{py:func}`parcels.convert.mitgcm_to_sgrid` and {py:func}`parcels.convert.croco_to_sgrid`). Each one uses what we know
about a particular model to rename variables and attach the SGRID metadata.

If your model is not covered, you can write your own convert function. This guide explains what Parcels expects,
walks through writing a converter for a made-up model, and shows how to check the result with `describe()`.

```{note}
This guide covers structured grids only (SGRID). Unstructured grids use the UGRID conventions and are not covered here.
```

## What a convert function needs to produce

A convert function takes raw model output and returns a single {py:class}`xarray.Dataset` that meets the
requirements below. The built-in converters take a `fields` dictionary (mapping the Parcels field name to a
`DataArray` or `Dataset`) and a `coords` dataset (holding the grid coordinates). We recommend that you use the same
signature.

| Requirement                                                                                                                       | Why Parcels needs it                                                                                                             |
| :-------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| A variable (conventionally named `grid`) with the attribute `cf_role="grid_topology"` and the SGRID attributes listed below.      | This is where Parcels learns the grid layout.                                                                                    |
| Every dimension of every data variable is either `time` or listed in the SGRID metadata.                                          | Parcels needs to know which axis (X, Y or Z) each dimension belongs to.                                                          |
| For time-varying fields: has a `time` coordinate of datetimes, timedeltas or `cftime` objects.                                    | Parcels uses it to interpolate in time and to check that the calendars of fields are compatible.                                 |
| The horizontal node (corner) coordinates are named `lon` and `lat`. They sit on the node dimensions and have a `units` attribute. | Particles are located using these coordinates. Units containing `degree` give a spherical mesh; anything else gives a flat mesh. |
| The vertical node coordinate is named `depth` and is strictly increasing.                                                         | Parcels uses `depth` for vertical search.                                                                                        |
| A `time` coordinate of datetimes, timedeltas or `cftime` objects.                                                                 | Parcels uses it to interpolate in time and to check that the calendars of fields are compatible.                                 |
| Velocities are named `U`, `V` and (optionally) `W`, with `W` positive in the same direction as positive depth.                    | Parcels auto-detects the `UV` and `UVW` vector fields from these names, which are then used in the built-in advection Kernels.   |

## SGRID in a nutshell

SGRID describes a staggered grid with two kinds of points along each axis:

- **nodes** are the cell corners. `lon` and `lat` must be defined here in Parcels.
- **faces** are the cell centres.

A note on the terminology used in this page. An **axis** is one of the physical directions of the model grid: X, Y and (for 3D data) Z. A
**dimension** is a named dimension of an array in the dataset (e.g. `jt` or `jq`). On a staggered grid,
variables sit at different positions along the same axis (e.g. Arakawa "F" and "ψ" points align with the nodes, while Arakawa "T" and "ρ" points align with the faces), so a single axis usually has more than one dimension.

SGRID keeps all fields and coordinates in a single dataset, so the dimensions along an axis must have distinct names.
For example, the Y-node dimension might be called `jq` and the Y-face dimension `jt`. Some models instead write
each field to a separate file and call both of these dimensions `y`, leaving it to the user to know where each field
sits on the grid. In that case, your convert function needs to rename the dimensions so that they are distinct.

On an Arakawa C-grid, the variables are placed as follows:

- the zonal velocity `U` sits on the X-node dimension and the Y-face dimension;
- the meridional velocity `V` sits on the X-face dimension and the Y-node dimension;
- tracers such as temperature sit on the face dimensions in both directions.

For each axis, SGRID pairs a face dimension with a node dimension and a **padding**. The padding says where the face
at index `i` lies relative to the node at index `i`:

```text
face:node (padding:none)
  ●─────●─────●─────●─────●
  0  0  1  1  2  2  3  3  4

face:node (padding:low)
  ─────●─────●─────●─────●─────●
    0  0  1  1  2  2  3  3  4  4

face:node (padding:high)
  ●─────●─────●─────●─────●─────
  0  0  1  1  2  2  3  3  4  4

face:node (padding:both)
  ─────●─────●─────●─────●─────●─────
    0  0  1  1  2  2  3  3  4  4  5
```

In these diagrams `●` is a node and `─────` is a face. The numbers are array indices. To choose the padding, start
by comparing the lengths of the face and node dimensions:

- If there is one face fewer than nodes, every face lies between two nodes and there are no extra faces at the ends
  of the domain: `padding:none`.
- If there is one face more than nodes, there is an extra face at both ends of the domain: `padding:both`.
- If the lengths are equal, there is an extra face at one end of the domain only. Face `i` then lies either on the
  low side of node `i` (`padding:low`) or on its high side (`padding:high`). **The array sizes cannot tell you which
  of the two is correct**, so you need to find this out from the model documentation.

The metadata is stored as string attributes on the `grid` variable:

| Attribute             | Example value                                 | Meaning                                                   |
| :-------------------- | :-------------------------------------------- | :-------------------------------------------------------- |
| `cf_role`             | `"grid_topology"`                             | Marks this variable as the grid description.              |
| `topology_dimension`  | `2`                                           | A 2D horizontal grid (vertical layers are handled below). |
| `node_dimensions`     | `"iq jq"`                                     | The X and Y node dimensions, in that order.               |
| `face_dimensions`     | `"it:iq (padding:high) jt:jq (padding:high)"` | `face:node (padding:...)` for X, then for Y.              |
| `vertical_dimensions` | `"kt:kw (padding:high)"`                      | `face:node (padding:...)` for Z (omit for 2D data).       |
| `node_coordinates`    | `"lon lat"`                                   | The variables holding the node longitude and latitude.    |

```{note}
Parcels represents 3D data as a 2D SGRID grid (`topology_dimension=2`) plus `vertical_dimensions`. SGRID grids with `topology_dimension=3` are not supported.
```

## Worked example: the "ToyOcean" model

Say we have output from a C-grid model called ToyOcean. Its documentation describes the grid as follows:

- `xq(iq)`, `yq(jq)`: longitude and latitude of the **south-west corner** of each cell.
- `xt(it)`, `yt(jt)`: longitude and latitude of each cell centre.
- `zw(kw)`: depth of the **top** of each layer. `zt(kt)`: depth of each layer's midpoint. Both are in metres, positive downward.
- `uvel(t, kt, jt, iq)`: eastward velocity on the western face of each cell.
- `vvel(t, kt, jq, it)`: northward velocity on the southern face of each cell.
- `wvel(t, kw, jt, it)`: vertical velocity on the top face of each cell, **positive upward**.
- `temp(t, kt, jt, it)`: temperature at the cell centre.

We'll create some synthetic ToyOcean output with a uniform flow. A uniform flow lets us check the final particle
position against the exact answer later on.

```{code-cell}
import numpy as np
import xarray as xr

import parcels

ni, nj, nk = 20, 15, 5
time = np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[ns]")
nt = time.size

xq = np.arange(ni, dtype=float)  # south-west corners
yq = 40.0 + np.arange(nj, dtype=float)
zw = np.linspace(0.0, 400.0, nk)  # layer tops

coords = xr.Dataset(
    {
        "xq": ("iq", xq, {"units": "degrees_east"}),
        "yq": ("jq", yq, {"units": "degrees_north"}),
        "xt": ("it", xq + 0.5, {"units": "degrees_east"}),
        "yt": ("jt", yq + 0.5, {"units": "degrees_north"}),
        "zw": ("kw", zw, {"units": "m", "positive": "down"}),
        "zt": ("kt", zw + 50.0, {"units": "m", "positive": "down"}),
    }
)


def field(name, dims, value):
    ds = xr.Dataset(
        {name: (dims, np.full((nt, nk, nj, ni), value))},
        coords={"t": time},
    )
    ds["t"].attrs['axis'] = 'T'
    return ds


uvel = field("uvel", ("t", "kt", "jt", "iq"), 0.1)  # 0.1 m/s eastward
vvel = field("vvel", ("t", "kt", "jq", "it"), 0.05)  # 0.05 m/s northward
wvel = field("wvel", ("t", "kw", "jt", "it"), 1e-5)  # 1e-5 m/s upward
temp = field("temp", ("t", "kt", "jt", "it"), 15.0)
```

### Step 1: Map the model grid onto SGRID

Before writing any code, fill in the face, node and padding for each axis:

| Axis | Face (centre) dim | Node (corner) dim | Padding | Reasoning                                                                                                      |
| :--- | :---------------- | :---------------- | :------ | :------------------------------------------------------------------------------------------------------------- |
| X    | `it`              | `iq`              | `high`  | Corner `i` is the western edge of cell `i`, so cell `i` lies between node `i` and node `i+1`.                  |
| Y    | `jt`              | `jq`              | `high`  | Corner `j` is the southern edge of cell `j`, so cell `j` lies between node `j` and node `j+1`.                 |
| Z    | `kt`              | `kw`              | `high`  | Depth increases downward and `zw[k]` is the top of layer `k`, so layer `k` lies between `zw[k]` and `zw[k+1]`. |

Then check the velocity positions against this table. `uvel` uses `iq` (X node) and `jt` (Y face), and `vvel` uses
`it` (X face) and `jq` (Y node). That is exactly the C-grid layout described above.

### Step 2: Write the convert function

The function below follows the same steps as the built-in converters:

1. extract the fields and give them their Parcels names;
2. merge them with the node coordinates only;
3. rename the coordinates to what Parcels expects;
4. fix sign conventions;
5. attach the SGRID metadata.

```{code-cell}
def toyocean_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset) -> xr.Dataset:
    """Create an SGRID-compliant xarray.Dataset from ToyOcean output."""
    # 1. Pull each field out as a DataArray named after its Parcels field name
    das = []
    for name, field in fields.items():
        da = field[next(iter(field.data_vars))] if isinstance(field, xr.Dataset) else field
        das.append(da.rename(name))

    # 2. Merge with only the node coordinates (Parcels doesn't need the cell-centre coordinates)
    ds = xr.merge([*das, coords[["xq", "yq", "zw"]]])
    ds.attrs.clear()

    # 3. Rename to the names that Parcels expects
    ds = ds.rename({"t": "time", "xq": "lon", "yq": "lat", "zw": "depth"})
    ds = ds.set_coords(["lon", "lat", "depth"])

    # 4. ToyOcean's W is positive upward but depth is positive downward; Parcels expects W to be in the direction of positive depth
    if "W" in ds:
        ds["W"] = -ds["W"]

    # 5. Attach the SGRID metadata
    ds["grid"] = xr.DataArray(
        0, # The value here doesn't matter - so we put 0. The attributes are the important aspect
        attrs={
            "cf_role": "grid_topology",
            "topology_dimension": 2,
            "node_dimensions": "iq jq",
            "face_dimensions": "it:iq (padding:high) jt:jq (padding:high)",
            "vertical_dimensions": "kt:kw (padding:high)",
            "node_coordinates": "lon lat",
        },
    )
    return ds


ds = toyocean_to_sgrid(
    fields={"U": uvel, "V": vvel, "W": wvel, "temperature": temp},
    coords=coords,
)
ds
```

```{tip}
If you need to rename a dimension _after_ attaching the `grid` variable, use `ds.sgrid.rename({...})` instead of
`ds.rename({...})`. It works the same way, but also updates the dimension names stored in the SGRID metadata.
```

### Step 3: Check the grid with `describe()`

The SGRID metadata can be read back through the `ds.sgrid` accessor. Calling `describe()` on it prints a summary of
the grid layout:

```{code-cell}
ds.sgrid.metadata.describe()
```

Go through the output and compare it with your table from Step 1:

- **The axis lines** (`X-axis`, `Y-axis`, `Z-axis`) should pair each face dimension with the correct node dimension
  and padding. If X and Y are swapped here, `node_dimensions` or `face_dimensions` is in the wrong order.
- **The staggered grid layout** shows where `u` (X-face) and `v` (Y-face) sit relative to the nodes `n`. Compare
  this with `ds["U"].dims` and `ds["V"].dims`: `U` should use the X node dimension and the Y face dimension, and
  `V` the other way round. On the vertical axis, `w` marks the depth levels (`depth`) that sit on the node
  dimension.
- **The axis padding diagrams** show which side of node `i` the face `i` sits on. Compare them with the grid
  figure in your model documentation.

```{important}
The padding is the most important check, because **Parcels cannot detect a wrong padding by itself**. With `low`
instead of `high`, the dataset still loads without any error, but every field is shifted by half a grid cell.
```

### Step 4: Create the FieldSet

Once the metadata looks right, pass the dataset to {py:func}`parcels.FieldSet.from_sgrid_conventions`. Use
{py:func}`parcels.FieldSet.describe` to check that the vector fields were found and that the mesh is what you expect:

```{code-cell}
fieldset = parcels.FieldSet.from_sgrid_conventions(ds)
fieldset.describe()
```

Here we check that:

- the `UV` and `UVW` vector fields are listed. If `UVW` is missing, the vertical velocity was not named `W`;
- the velocity interpolator is `CGrid_Velocity`. Parcels picks this when `U` and `V` have different dimensions, so
  `XLinear_Velocity` would mean that Parcels treats the data as an A-grid;
- the mesh is `SphericalMesh`, because the `lon` units contain "degree".

### Step 5: Check the result with a short run

It is worth checking the new converter with a short run where you know the answer. With our uniform flow, one day
of advection should move a particle $0.1 \times 86400$ m east, $0.05 \times 86400$ m north, and $10^{-5} \times 86400$ m
_up_. The last check also confirms that the sign of `W` was flipped correctly.

```{code-cell}
pset = parcels.ParticleSet(fieldset, x=[5.0], y=[45.0], z=[100.0])
pset.execute(
    parcels.kernels.AdvectionRK4_3D,
    runtime=np.timedelta64(1, "D"),
    dt=np.timedelta64(1, "h"),
    verbose_progress=False,
)

seconds = 86400
m_per_deg = 1852 * 60
expected_x = 5.0 + 0.1 * seconds / (m_per_deg * np.cos(np.deg2rad(45.0)))
expected_y = 45.0 + 0.05 * seconds / m_per_deg
expected_z = 100.0 - 1e-5 * seconds

print(f"x: {pset.x[0]:.4f} (expected ~{expected_x:.4f})")
print(f"y: {pset.y[0]:.4f} (expected ~{expected_y:.4f})")
print(f"z: {pset.z[0]:.4f} (expected ~{expected_z:.4f})")

np.testing.assert_allclose(pset.x[0], expected_x, atol=1e-3)
np.testing.assert_allclose(pset.y[0], expected_y, atol=1e-3)
np.testing.assert_allclose(pset.z[0], expected_z, atol=1e-3)
```

## Troubleshooting

These are the most common problems when writing a converter, together with the error that each one causes:

| Error message (abbreviated)                                                                    | Cause                                                                                                        | Fix                                                                                                                                            |
| :--------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- |
| `No variable found in dataset with 'cf_role' attribute set to 'grid_topology'`                 | No SGRID metadata has been attached.                                                                         | Add the `grid` variable (Step 2, part 5).                                                                                                      |
| `DataArray 'U' with dims (...) has dimensions {'kt'} that are not associated with a direction` | A field dimension is not listed in the SGRID metadata.                                                       | Add it to `face_dimensions` or `vertical_dimensions`, or drop the extra dimension (e.g. with `.isel`).                                         |
| `No variable named 'lat'`                                                                      | The node coordinates have not been renamed.                                                                  | Rename them to `lon` and `lat`, and make sure `node_coordinates` is `"lon lat"`.                                                               |
| `Coordinate 'lon' of your dataset has no 'units' attribute`                                    | Parcels cannot tell whether the mesh is spherical or flat.                                                   | Set `.attrs['units']` (e.g. `"degrees_east"`or`"m"`) on your Xarray coordinate arrays accordingly, or pass `mesh=`to`from_sgrid_conventions`.  |
| `Depth DataArray 'depth' ... must be strictly increasing`                                      | Depth is stored as negative values, or ordered from the bottom up.                                           | Negate `depth` if it is negative. If the levels run bottom-up, reverse both vertical dimensions and swap `high`/`low` in the vertical padding. |
| `Expected right to be a np.timedelta64, datetime, cftime.datetime, or np.datetime64`           | `time` is stored as plain numbers.                                                                           | Decode it (e.g. with `xr.decode_cf`), or convert it to `timedelta64`.                                                                          |
| _No error, but particles are shifted by half a cell_                                           | Wrong padding.                                                                                               | Compare the `describe()` padding diagrams with the grid figure in your model documentation.                                                    |
| _No error, but particles don't move vertically, or move the wrong way_                         | Vertical velocity is not named `W`, or its sign convention is different from the sign convention of `depth`. | Rename it to `W`, and negate it to align with the positive `depth` direction.                                                                  |

## A template to start from

```python
def mymodel_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset) -> xr.Dataset:
    das = []
    for name, field in fields.items():
        da = field[next(iter(field.data_vars))] if isinstance(field, xr.Dataset) else field
        das.append(da.rename(name))

    ds = xr.merge([*das, coords[["<x_node_coord>", "<y_node_coord>", "<z_node_coord>"]]])
    ds.attrs.clear()

    ds = ds.rename({
        "<time_dim>": "time",
        "<x_node_coord>": "lon",
        "<y_node_coord>": "lat",
        "<z_node_coord>": "depth",
    })
    ds = ds.set_coords(["lon", "lat", "depth"])

    # Model-specific fixes go here, e.g. sign of W, depth direction, time decoding, units

    ds["grid"] = xr.DataArray(
        0,
        attrs={
            "cf_role": "grid_topology",
            "topology_dimension": 2,
            "node_dimensions": "<x_node_dim> <y_node_dim>",
            "face_dimensions": "<x_face_dim>:<x_node_dim> (padding:<p>) <y_face_dim>:<y_node_dim> (padding:<p>)",
            "vertical_dimensions": "<z_face_dim>:<z_node_dim> (padding:<p>)",
            "node_coordinates": "lon lat",
        },
    )
    return ds
```

Then check the result:

```python
ds = mymodel_to_sgrid(fields={...}, coords=...)
ds.sgrid.metadata.describe()
fieldset = parcels.FieldSet.from_sgrid_conventions(ds)
fieldset.describe()
```

## Contributing your converter to Parcels

If other people use your model too, please consider contributing your converter to Parcels. Inside
`src/parcels/convert.py`, converters build the metadata with Parcels' internal SGRID helpers rather than raw strings.
The helpers validate the structure when it is created:

```python
import parcels._sgrid as sgrid

ds["grid"] = xr.DataArray(
    0,
    attrs=sgrid.SGrid2DMetadata(
        cf_role="grid_topology",
        topology_dimension=2,
        node_dimensions=("iq", "jq"),
        node_coordinates=("lon", "lat"),
        face_dimensions=(
            sgrid.FaceNodePadding("it", "iq", sgrid.Padding.HIGH),
            sgrid.FaceNodePadding("jt", "jq", sgrid.Padding.HIGH),
        ),
        vertical_dimensions=(sgrid.FaceNodePadding("kt", "kw", sgrid.Padding.HIGH),),
    ).to_attrs(),
)
```

A contribution would typically include the function in `convert.py`, a test in `tests/test_convert.py`, and a
tutorial notebook in the "Converting model data to FieldSets" section of this user guide. See the
[development guide](../../development/index.md) for how to get started.
