---
file_format: mystnb
kernelspec:
  name: python3
---

# 📖 Parcels conceptual workflow

Parcels is a set of Python classes and methods to create customisable particle tracking simulations using gridded output from (ocean) circulation models.

Here, we will explain the most important classes and functions. This overview can be useful to start understanding the different components we use in Parcels, and to structure the code in a simulation script.

A Parcels simulation is generally built up from four different components:

1. [**FieldSet**](#1-fieldset). The input dataset of gridded fields (e.g. ocean current velocity, temperature) in which virtual particles are defined.
2. [**ParticleSet**](#2-particleset). The dataset of virtual particles. These always contain t, z, y, and x, for which initial values must be defined. The ParticleSet may also contain other, custom variables.
3. [**Kernels**](#3-kernels). Kernels perform some specific operation on the particles every time step (e.g. advect the particles with the three-dimensional flow; or interpolate the temperature field to the particle location).
4. [**Execute**](#4-execute). Execute the simulation. The core method which integrates the operations defined in Kernels for a given runtime and timestep, and writes output to a ParticleFile.

We discuss each component in more detail below. The subsections titled **"Learn how to"** link to more detailed [how-to guide notebooks](../index.md) and more detailed _explanations_ of Parcels functionality are included under **"Read more about"** subsections. The full list of classes and methods is in the [API reference](../../reference/parcels/index). If you want to learn by doing, check out the [quickstart tutorial](./tutorial_quickstart.md) to start creating your first Parcels simulation.

```{figure} ../../_static/concepts_diagram.png
:alt: Parcels concepts diagram
:width: 100%

Parcels concepts diagram with key classes in blue boxes
```

## 1. FieldSet

Parcels provides a framework to simulate particles **within a set of fields**, such as flow velocities and temperature. To start a Parcels simulation we must define this dataset with the **{py:obj}`parcels.FieldSet`** class.

The input dataset from which to create a {py:obj}`parcels.FieldSet` can be an [`xarray.Dataset`](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dataset) with output from a hydrodynamic model or reanalysis. Such a dataset usually contains a number of gridded variables (e.g. `"U"`), which in Parcels become {py:obj}`parcels.Field` objects. A set of {py:obj}`parcels.Field` objects is stored in a {py:obj}`parcels.FieldSet` in an analoguous way to how `xarray.DataArray` objects combine to make an `xarray.Dataset`.

For several common input datasets, such as the Copernicus Marine Service analysis products, Parcels has a specific method to read and parse the data correctly:

```python
ds_fields = xr.open_mfdataset("insert_copernicus_data_files.nc")
fields = {"U": ds_fields["uo"], "V": ds_fields["vo"]}
ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)
```

In some cases, we might want to combine fields from different sources in the same {py:obj}`parcels.FieldSet`, such as ocean currents from one dataset and Stokes drift from another. This is possible in Parcels by creating multiple {py:obj}`parcels.FieldSet` objects and combining them into a single {py:obj}`parcels.FieldSet`:

```python
ds_fields2 = xr.open_dataset("insert_stokes_data_files.nc")
fields2 = {"Ustokes": ds_fields2["ustokes"], "Vstokes": ds_fields2["vstokes"]}
ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields2)
fieldset += parcels.FieldSet.from_sgrid_conventions(
    ds_fset, vector_fields={"UVstokes": ["Ustokes", "Vstokes"]}
)
```

### Grid

Each {py:obj}`parcels.Field` is defined on a grid. With Parcels, we can simulate particles in fields on both structured (**{py:obj}`parcels.XGrid`**) and unstructured (**{py:obj}`parcels.UxGrid`**) grids. The grid is defined by the coordinates of grid cell nodes, edges, and faces. {py:obj}`parcels.XGrid` objects are based on Xarray Datasets with attached SGRID metadata, while {py:obj}`parcels.UxGrid` objects are based on [`uxarray.Grid`](https://uxarray.readthedocs.io/en/stable/generated/uxarray.Grid.html#uxarray.Grid) objects.

The user doesn't need to manually construct or manage grids - this is done internally by Parcels. When using {py:obj}`parcels.FieldSet.from_sgrid_conventions()` or {py:obj}`parcels.FieldSet.from_ugrid_conventions()` the correct grid object is constructed for your data.

```{admonition} 📖 Read more about grids
:class: seealso
- [Grids explanation](../examples/explanation_grids.ipynb)
```

### Interpolation

To find the value of a {py:obj}`parcels.Field` at any particle location, Parcels interpolates the gridded field. Depending on the variable, grid, and required accuracy, different interpolation methods may be appropriate. Parcels comes with a number of built-in **{py:obj}`parcels.interpolators`**.

```{admonition} 📖 Read more about interpolation
:class: seealso
- [Interpolation explanation](../examples/explanation_interpolation.md)
```

```{admonition} 🖥️ Learn how to use Parcels interpolators
:class: seealso
- [Interpolators guide](../examples/tutorial_interpolation.ipynb)
```

## 2. ParticleSet

Once the environment has a {py:obj}`parcels.FieldSet` object, you can start defining your particles in a **{py:obj}`parcels.ParticleSet`** object. This object requires:

1. The {py:obj}`parcels.FieldSet` object in which the particles will be released.
2. The type of {py:obj}`parcels.Particle`: A default `Particle` or a custom `Particle`-type with additional {py:obj}`parcels.Variable`s (see the [custom kernel example](custom-kernel)).
3. Initial conditions for each {py:obj}`parcels.Variable` defined in the {py:obj}`parcels.Particle`, most notably the release coordinates of `t`, `z`, `y` and `x`.

```python
t = np.array([0])
z = np.array([0])
y = np.array([0])
x = np.array([0])

# Create a ParticleSet
pset = parcels.ParticleSet(
    fieldset=fieldset, pclass=parcels.Particle, t=t, z=z, y=y, x=x
)
```

```{admonition} 🖥️ Learn more about how to create ParticleSets
:class: seealso
- [Release particles at different times](../examples/tutorial_delaystart.ipynb)
- [Sampling initial Field values](../examples/tutorial_sampling.ipynb#sampling-initial-field-values)
```

## 3. Kernels

A **`kernel`** function is a little snippet of code, which is applied to the particles in the {py:obj}`parcels.ParticleSet`, for every time step during a simulation. Kernels define the computation or numerical integration done by Parcels, and can represent many processes such as advection, ageing, growth, or simply the sampling of a field.

Advection of a particle by the flow, the change in position $\mathbf{x}(t) = (x(t), y(t))$ at time $t$, can be described by the equation:

$$
\begin{aligned}
\frac{\text{d}\mathbf{x}(t)}{\text{d}t} = \mathbf{v}(\mathbf{x}(t),t),
\end{aligned}
$$

where $\mathbf{v}(\mathbf{x},t) = (u(\mathbf{x},t), v(\mathbf{x},t))$ describes the ocean velocity field at position $\mathbf{x}$ at time $t$.

In Parcels, we can write a kernel function which integrates this equation at each timestep `particles.dt`. To do so, we need the ocean velocity field `fieldset.UV` at the `particles` location, and compute the change in position, `particles.dx` and `particles.dy`.

```python
@parcels.validate_kernel
def AdvectionEE(particles, fieldset):
    """Advection of particles using Explicit Euler (aka Euler Forward) integration."""
    u1, v1 = fieldset.UV[particles]
    particles.dx += u1 * particles.dt
    particles.dy += v1 * particles.dt
```

Note that the (optional) `@parcels.validate_kernel` decorator can be used to ensure that the kernel function is correctly defined and follows recommendations.

Basic kernels are included in Parcels to compute advection and diffusion. The standard advection kernel is {py:obj}`parcels.kernels.AdvectionRK2`, a [second-order Runge-Kutta integrator](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method) of the advection function.

```{warning}
It is advised _not_ to update the particle coordinates (`particles.t`, `particles.z`, `particles.y`, or `particles.x`) directly within a Kernel, as that can negatively interfere with the way that particle movements by different kernels are vectorially added. Use a change in the coordinates: `particles.dy`, `particles.dx` and/or `particles.dz`. Read the [kernel loop tutorial](../examples/explanation_kernelloop.md) to understand why.
```

(custom-kernel)=
We can write custom kernels to add many different types of 'behaviour' to the particles. To do so, we write a function with two arguments: `particles` and `fieldset`. We can then write any computation as a function of any variables defined in the `Particle` and any `Field` defined in the `FieldSet`. Kernels can then be combined by creating a `list` of the kernels. The kernels are executed in order:

```python
# Create a custom Particle object with an "age" variable
CustomParticle = parcels.Particle.add_variable(parcels.Variable("age", initial=0))


# Create a custom kernel which keeps track of the particle age
def Age(particles, fieldset):
    particles.age += particles.dt


# define all kernels to be executed on particles using an (ordered) list
kernels = [Age, parcels.kernels.AdvectionRK2]
```

```{note}
Every Kernel must be a function with the following (and only those) arguments: `(particles, fieldset)`
```

```{warning}
We have to be careful with kernels that sample velocities on spherical grids (so with longitude and latitude in degrees). Parcels can automatically convert velocities from m s<sup>-1</sup> to degrees s<sup>-1</sup>, but only when using `VectorFields`. [This guide](../examples/tutorial_velocityconversion.ipynb) describes how to use velocities on a spherical grid in Parcels.
```

```{admonition} 📖 Read more about the Kernel loop
:class: seealso
- [The Kernel loop](../examples/explanation_kernelloop.md)
```

```{admonition} 🖥️ Learn how to write Kernels
:class: seealso
- [Sample fields like temperature](../examples/tutorial_sampling.ipynb).
- [Mimic the behaviour of ARGO floats](../examples/tutorial_Argofloats.ipynb).
- [Add diffusion to approximate subgrid-scale processes and unresolved physics](../examples/tutorial_diffusion.ipynb).
- [Convert velocities between units in m s<sup>-1</sup> and degrees s<sup>-1</sup>](../examples/tutorial_velocityconversion.ipynb).
```

## 4. Execute

The execution of the simulation is done using the method **{py:func}`parcels.ParticleSet.execute()`**, given the `FieldSet`, `ParticleSet`, and `Kernels` defined in the previous steps. This method requires the following arguments:

1. The kernels to be executed.
2. The `runtime` defining how long the execution loop runs. Alternatively, you may define the `endtime` at which the execution loop stops.
3. The timestep `dt` at which to execute the kernels.
4. (Optional) The `ParticleFile` object to write the output to.

```python
dt = np.timedelta64(5, "m")
runtime = np.timedelta64(1, "D")

# Run the simulation
pset.execute(kernels=kernels, dt=dt, runtime=runtime)
```

### Output

To analyse the particle data generated in the simulation, we need to define a {py:obj}`parcels.ParticleFile` and add it as an argument to {py:func}`parcels.ParticleSet.execute()`. The output will be written in a [parquet format](https://parquet.apache.org/), which can be opened as a `polars.DataFrame`. The dataset will contain the particle data with at least `t`, `z`, `y` and `x`, for each particle at timesteps defined by the `outputdt` argument.

There are many ways to analyze particle output, and although we provide [a short tutorial to get started](./tutorial_output.ipynb), we recommend writing your own analysis code and checking out [related Lagrangian analysis projects in our community page](../../community/index.md#analysis-code).

```{admonition} 🖥️ Learn how to run a simulation
:class: seealso
- [Choose an appropriate timestep and integrator](../examples/tutorial_dt_integrators.ipynb)
- [Work with Parcels output](./tutorial_output.ipynb)
```
