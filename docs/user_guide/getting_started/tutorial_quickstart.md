---
file_format: mystnb
kernelspec:
  name: python3
---

# 🎓 Quickstart tutorial

Welcome to the Parcels quickstart tutorial, in which we will go through all the necessary steps to run a simulation.
The code in this notebook can be used as a starting point to run Parcels in your own environment. Along the way we will
familiarize ourselves with some specific classes and methods. If you are ever confused about one of these and want to
read more, we have a [concepts overview](./explanation_concepts.md) discussing them in more detail. Let's dive in!

## Imports

Parcels depends on `xarray`, expecting inputs in the form of [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html). Output files can be read with [Polars](https://pola.rs).

```{code-cell}
import numpy as np
import xarray as xr
import polars as pl
import parcels
import parcels.tutorial
```

## Input flow fields: `FieldSet`

A Parcels simulation of Lagrangian trajectories of virtual particles requires two inputs; the first is a set of
hydrodynamics fields in which the particles are tracked. Here we provide an example using a subset of the
[Global Ocean Physics Reanalysis](https://doi.org/10.48670/moi-00021) from the Copernicus Marine Service.

This dataset is bundled in Parcels' tutorial data registry. {py:func}`parcels.tutorial.open_dataset()` downloads the
files into a local cache on first use (subsequent calls return the cached copy) and opens them as an `xarray` dataset.
Use {py:func}`parcels.tutorial.list_datasets()` to see all available tutorial datasets.

```{code-cell}
print(
  "Datasets here are cached at",
  parcels.tutorial.cache_path ,
  " . Note that this varies machine to machine.",
) # manually delete this folder to force re-downloads
ds_fields = parcels.tutorial.open_dataset("CopernicusMarine_data_for_Argo_tutorial/data")
ds_fields
```

As we can see, the reanalysis dataset contains eastward velocity `uo`, northward velocity `vo`, potential temperature
(`thetao`) and salinity (`so`) fields.

These hydrodynamic fields need to be stored in a {py:obj}`parcels.FieldSet` object. Parcels provides tooling to parse many types
of models or observations into such a `parcels.FieldSet` object. This is done in a two-step approach.

First, we convert the dataset into an SGRID-compliant dataset, for example by using a version of `parcels.convert.<MODEL>_to_sgrid()`. Then, we create the {py:obj}`parcels.FieldSet` from the SGRID-compliant dataset using {py:func}`parcels.FieldSet.from_sgrid_conventions()`.

Below, we use a combination of {py:func}`parcels.convert.copernicusmarine_to_sgrid()` and {py:func}`parcels.FieldSet.from_sgrid_conventions()`, providing the names of the velocity fields in the dataset in the dictionary `fields`:

```{code-cell}
fields = {"U": ds_fields["uo"], "V": ds_fields["vo"]}
ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)
```

Now, in order to improve performance, we can convert the {py:obj}`parcels.FieldSet` to windowed arrays. This is especially useful for large datasets with many timeslices, as it allows Parcels to load only the necessary timeslices into memory during the simulation. For more information about squeezing performance out of Parcels, see the [performance tutorial](../examples/explanation_performance.md).

```{code-cell}
# Convert the FieldSet to windowed arrays for better performance
fieldset = fieldset.to_windowed_arrays()
```

You can inspect the {py:obj}`parcels.FieldSet` object with the `describe` method in order to see which {py:obj}`parcels.Field`s are included, and which grid and interpolation method is used for each field. This also gives information on the type of mesh and the time interval of the {py:obj}`parcels.FieldSet`:

```{code-cell}
fieldset.describe()
```

The subset contains a region of the Agulhas current along the southeastern coast of Africa:

```{code-cell}
temperature = ds_fields.isel(time=0, depth=0).thetao.plot(cmap="magma")
velocity = ds_fields.isel(time=0, depth=0).plot.quiver(x="longitude", y="latitude", u="uo", v="vo")
```

## Input virtual particles: `ParticleSet`

Now that we have created a {py:obj}`parcels.FieldSet` object from the hydrodynamic data, we need to provide our second input:
the virtual particles for which we will calculate the trajectories.

We need to create a {py:obj}`parcels.ParticleSet` object with the particles' initial time and position. The {py:obj}`parcels.ParticleSet`
object also needs to know about the {py:obj}`parcels.FieldSet` in which the particles "live". Finally, we need to specify the type of
{py:obj}`parcels.ParticleClass` we want to use. The default particles have `t`, `z`, `y`, and `x`, but you can easily add
other {py:obj}`parcels.Variable`s such as size, temperature, or age to create your own particles to mimic plastic or an [ARGO float](../examples/tutorial_Argofloats.ipynb).

```{code-cell}
# Particle locations and initial time
npart = 10  # number of particles to be released
# release particles in a line along a meridian
lat = np.linspace(-32.5, -30.5, npart)
lon = np.repeat(32, npart)
time = np.repeat(ds_fields.time.values[0], npart) # at initial time of input data
z = np.repeat(0.5, npart) # at 0.5 meter depth (first level of the input data)

pset = parcels.ParticleSet(
    fieldset=fieldset, pclass=parcels.Particle, t=time, z=z, y=lat, x=lon
)
```

And you can plot the particles on top of the temperature and velocity field:

```{code-cell}
temperature = ds_fields.isel(time=0, depth=0).thetao.plot(cmap="magma")
velocity = ds_fields.isel(time=0, depth=0).plot.quiver(x="longitude", y="latitude", u="uo", v="vo")
ax = temperature.axes
ax.scatter(lon, lat, s=40, c='w', edgecolors='r');
```

If you also want to sample the initial value of a field, you can do so by calling the field with the {py:obj}`parcels.ParticleSet` as an argument. For example, in the code above you would add one line (assuming that the ParticleSet has a variable `temperature`) to sample the initial temperature values, before calling `pset.execute(...)`.

```python
pset.temperature = fieldset.thetao[pset]
```

See the [sampling tutorial](../examples/tutorial_sampling.ipynb#sampling-initial-field-values) for more information on initial sampling of fields.

## Compute: `Kernel`

After setting up the input data and particle start locations and times, we need to specify what calculations to do with
the particles. These calculations, or numerical integrations, will be performed by what we call a `kernel` function, operating on
all particles in the `ParticleSet`. The most common calculation is the advection of particles through the velocity field.
Parcels comes with a number of common {py:obj}`parcels.kernels`, from which we will use the Runge-Kutta advection kernel {py:obj}`parcels.kernels.AdvectionRK2`:

```{code-cell}
kernels = [parcels.kernels.AdvectionRK2]
```

## Prepare output: `ParticleFile`

Before starting the simulation, we must define where and how frequent we want to write the output of our simulation.
We can define this in a {py:obj}`parcels.ParticleFile` object:

```{code-cell}
output_file = parcels.ParticleFile("output-quickstart.parquet", outputdt=np.timedelta64(1, "h"))
```

The output files are in `.parquet` [format](https://parquet.apache.org/), which can be read by [Polars](https://pola.rs/).
See the [Parcels output tutorial](./tutorial_output.ipynb) for more information on the parquet format. We want to choose
the `outputdt` argument so that it captures the smallest timescales of our interest.

## Run Simulation: `ParticleSet.execute()`

Finally, we can run the simulation by _executing_ the `ParticleSet` using the specified list of `kernels`. This is done using the {py:meth}`parcels.ParticleSet.execute()` method.
Additionally, we need to specify:

- the `runtime`: for how long we want to simulate particles.
- the `dt`: the timestep with which to perform the numerical integration in the `kernels`. Depending on the numerical
  integration scheme, the accuracy of our simulation will depend on `dt`. Read [this notebook](https://github.com/Parcels-code/10year-anniversary-session2/blob/8931ef69577dbf00273a5ab4b7cf522667e146c5/advection_and_windage.ipynb)
  to learn more about numerical accuracy.

```{code-cell}
:tags: [hide-output]
pset.execute(
    kernels,
    runtime=np.timedelta64(1, "D"),
    dt=np.timedelta64(5, "m"),
    output_file=output_file,
)
```

## Read output

To start analysing the trajectories computed by Parcels, we can open the {py:obj}`parcels.ParticleFile` using the {py:func}`parcels.read_particlefile()` utility, which itself uses `polars`:

```{code-cell}
df = parcels.read_particlefile("output-quickstart.parquet")
df
```

The file contains 250 rows: 25 observations for the 10 particle trajectories.
The [working with Parcels output tutorial](./tutorial_output.ipynb) provides more detail about the dataset and how to analyse it.

Let's verify that Parcels has computed the advection of the virtual particles!

```{code-cell}
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# convert time to matplotlib date format for plotting
time_values = mdates.date2num(df["t"].to_list())

# plot positions and colour particles by time
scatter = plt.scatter(df['x'], df['y'], c=time_values)
plt.scatter(df['x'][:npart], df['y'][:npart], facecolors="none", edgecolors='r') # starting positions
plt.scatter(lon, lat, facecolors="none", edgecolors='r') # starting positions
plt.xlim(31,33)
plt.ylabel("Latitude [deg N]")
plt.ylim(-33,-30)
colorbar = plt.colorbar(scatter, label="Time")
colorbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
plt.show()
```

That looks good! The virtual particles released in a line along the 32nd meridian (dark blue) have been advected by the
flow field.

## Running a simulation backwards in time

Now that we know how to run a simulation, we can easily run another and change one of the settings. We can trace back
the particles from their current to their original position by running the simulation backwards in time. To do so, we
can simply make `dt` < 0.

```{note}
We have not edited the `ParticleSet`, which means that the new simulation will start with the particles at their current
location!
```

```{code-cell}
:tags: [hide-output]
# set up output file
output_file = parcels.ParticleFile("output-backwards.parquet", outputdt=np.timedelta64(1, "h"))

# execute simulation in backwards time
pset.execute(
    kernels,
    runtime=np.timedelta64(1, "D"),
    dt=-np.timedelta64(5, "m"),
    output_file=output_file,
)
```

When we check the output, we can see that the particles have returned to their original position!

```{code-cell}
df_back = parcels.read_particlefile("output-backwards.parquet")

# convert time to matplotlib date format for plotting
time_values = mdates.date2num(df_back["t"].to_list())

scatter = plt.scatter(df_back['x'], df_back['y'], c=time_values)
particles_at_max_time = df_back.filter(pl.col("t") == df_back["t"].max())
plt.scatter(particles_at_max_time['x'], particles_at_max_time['y'], facecolors="none", edgecolors='r') # starting positions
plt.xlabel("Longitude [deg E]")
plt.xlim(31,33)
plt.ylabel("Latitude [deg N]")
plt.ylim(-33,-30)
colorbar = plt.colorbar(scatter, label="Time")
colorbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
plt.show()
```

Using RK2 advection, the final positions are equal to the original positions with an accuracy of 5 decimals:

```{code-cell}
# testing that final location == original location
particles_at_min_time = df_back.filter(pl.col("t") == df_back["t"].min())
np.testing.assert_almost_equal(particles_at_min_time["y"], lat, decimal=5)
np.testing.assert_almost_equal(particles_at_min_time['x'], lon, decimal=5)
```
