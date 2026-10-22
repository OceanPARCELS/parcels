---
file_format: mystnb
kernelspec:
  name: python3
---

# 🖥️ Working with Status Codes

In order to capture errors in the [Kernel loop](explanation_kernelloop.md), Parcels uses a Status Code system. There are several Status Codes, listed below.

```{code-cell}
import parcels

for statuscode, val in parcels.StatusCode.__dict__.items():
    if statuscode.startswith("__"):
        continue
    print(f"{statuscode} = {val}")
```

Once an error is thrown (for example, a Field Interpolation error), then the `particles.state` is updated to the corresponding status code. This gives you the flexibility to write a Kernel that checks for a status code and does something with it.

For example, you can write a Kernel that checks for `particles.state == parcels.StatusCode.ErrorOutOfBounds` and deletes the particle, and then append this custom Kernel to the Kernel list in `pset.execute()`.

```
def DeleteOutOfBounds(particles, fieldset):
    out_of_bounds = particles.state == parcels.StatusCode.ErrorOutOfBounds
    particles[out_of_bounds].state = parcels.StatusCode.Delete


def DeleteAnyError(particles, fieldset):
    any_error = particles.state >= 50  # This captures all Errors
    particles[any_error].state = parcels.StatusCode.Delete
```

But of course, you can also write code for more sophisticated behaviour than just deleting the particle. It's up to you! Note that if you don't delete the particle, you will have to update the `particles.state = parcels.StatusCode.Success` yourself. For example:

```
def Move1DegreeWest(particles, fieldset):
    out_of_bounds = particles.state == parcels.StatusCode.ErrorOutOfBounds
    particles[out_of_bounds].dx -= 1.0
    particles[out_of_bounds].state = parcels.StatusCode.Success
```

Or, if you want to make sure that particles don't escape through the water surface

```{code-cell}
def KeepInOcean(particles, fieldset):
    # find particles that move through the surface
    through_surface = particles.state == parcels.StatusCode.ErrorThroughSurface

    # move particles to surface
    particles[through_surface].dz = fieldset.surface - particles[through_surface].z

    # change state from error to success
    particles[through_surface].state = parcels.StatusCode.Success
```

Kernel functions such as the ones above can then be added to the list of kernels in `pset.execute()`.

Let's add the `KeepInOcean` Kernel to an particle simulation where particles move through the surface:

```{code-cell}
import numpy as np
from parcels._datasets.structured.generated import simple_UV_dataset

ds = simple_UV_dataset(n_times=1, nz=2, ny=5, nx=4, mesh="flat")

dx, dy = 1.0 / len(ds.XG), 1.0 / len(ds.YG)

# Add W velocity that pushes through surface
ds["W"] = ds["U"] - 0.1 # 0.1 m/s towards the surface

fieldset = parcels.FieldSet.from_sgrid_conventions(ds)
fieldset.add_context("surface", 0)  # surface is at z=0
```

If we advect particles with the `AdvectionRK2_3D` kernel, Parcels will raise a `FieldOutOfBoundSurfaceError`:

```{code-cell}
:tags: [raises-exception]
pset = parcels.ParticleSet(fieldset, parcels.Particle, z=[0.5], y=[2], x=[1.5])
kernels = [parcels.kernels.AdvectionRK2_3D]
pset.execute(kernels, runtime=np.timedelta64(1, "m"), dt=np.timedelta64(1, "s"), verbose_progress=False)
```

When we add the `KeepInOcean` Kernel, particles will stay at the surface:

```{code-cell}
pset = parcels.ParticleSet(fieldset, parcels.Particle, z=[0.5], y=[2], x=[1.5])

kernels = [parcels.kernels.AdvectionRK2_3D, KeepInOcean]

pset.execute(kernels,runtime=np.timedelta64(20, "s"), dt=np.timedelta64(1, "s"), verbose_progress=False)

print(f"particle z at end of run = {pset.z}")
```

```{note}
Kernels that control what to do with `particles.state` should typically be added at the _end_ of the Kernel list, because otherwise later Kernels may overwrite the `particles.state` or the `particles.dx` variables (see [Kernel loop explanation](explanation_kernelloop.md)).
```
