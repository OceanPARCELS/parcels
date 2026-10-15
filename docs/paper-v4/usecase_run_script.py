import glob
import os

import copernicusmarine
import numpy as np
import uxarray as ux
import xarray as xr

import parcels

DIR = "/storage/shared/oceanparcels/input_data/MatroosWaddenSea/DCSMv7_harmonie"

XR_MFDATASET_OPTIONS = {
    "combine": "nested",
    "concat_dim": "time",
    "data_vars": "minimal",
    "coords": "minimal",
    "compat": "override",
    "join": "override",
    "parallel": True,
    "chunks": {"time": 1},
}

# %% Open flow files
files = sorted(glob.glob(f"{DIR}/flow/dcsm_fm100m_harmonie_*"))

ds = xr.open_mfdataset(
    files,
    **XR_MFDATASET_OPTIONS,
)

uxgrid = ux.Grid.from_topology(
    node_lon=ds["Mesh_node_x"],
    node_lat=ds["Mesh_node_y"],
    face_node_connectivity=ds["tri_face_nodes"],
    fill_value=-1,
)

uxds = ux.UxDataset(
    xr.Dataset(
        {"U": ds["U"], "V": ds["V"]},
        coords={
            "zf": ("zf", ds["zf"].data),
            "zc": ("zc", ds["zc"].data),
            "time": ds["time"],
        },
    ),
    uxgrid=uxgrid,
)

fieldset = parcels.FieldSet.from_ugrid_conventions(uxds, mesh="spherical")

# %% Add Stokes drift to the fieldset
files = sorted(glob.glob(f"{DIR}/waves/swan_kuststrook_harmonie_*.nc"))

ds = xr.open_mfdataset(
    files,
    **XR_MFDATASET_OPTIONS,
)
Us = 2 * np.pi**3 * ds["wave_height_hm0"] ** 2 / (ds["wave_period_tm10"] ** 3 * 9.81)
ds["Us"] = Us * np.cos(ds["wave_dir_th0"] * np.pi / 180)
ds["Vs"] = Us * np.sin(ds["wave_dir_th0"] * np.pi / 180)

ds = ds[["lon", "lat", "Us", "Vs"]].expand_dims(dim={"depth": [0]})

for var in ["lon", "lat"]:
    ds[var] = ds[var].transpose("col", "row")

ds["grid"] = xr.DataArray(
    0,
    attrs=parcels._sgrid.SGrid2DMetadata(
        cf_role="grid_topology",
        topology_dimension=2,
        node_dimensions=("row", "col"),
        node_coordinates=("lon", "lat"),
        face_dimensions=(
            parcels._sgrid.FaceNodePadding("X", "row", parcels._sgrid.Padding.LOW),
            parcels._sgrid.FaceNodePadding("Y", "col", parcels._sgrid.Padding.LOW),
        ),
        vertical_dimensions=(
            parcels._sgrid.FaceNodePadding("Z", "depth", parcels._sgrid.Padding.HIGH),
        ),
    ).to_attrs(),
)

fieldset_waves = parcels.FieldSet.from_sgrid_conventions(
    ds, vector_fields={"UVStokes": ("Us", "Vs")}
)
fieldset += fieldset_waves


# %% Add wind to the fieldset

ds = copernicusmarine.open_dataset(
    dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
    variables=["eastward_wind", "northward_wind"],
    minimum_longitude=1,
    maximum_longitude=8,
    minimum_latitude=51,
    maximum_latitude=55,
    start_datetime="2025-11-01T00:00:00",
    end_datetime="2025-12-01T00:00:00",
)
ds = parcels.convert.copernicusmarine_to_sgrid(
    fields={
        "eastward_wind": ds["eastward_wind"],
        "northward_wind": ds["northward_wind"],
    }
)
fieldset_wind = parcels.FieldSet.from_sgrid_conventions(
    ds, vector_fields={"UVWind": ("eastward_wind", "northward_wind")}
)
fieldset += fieldset_wind

fieldset = fieldset.to_chunk_cached_arrays()
fieldset.describe()

# %% Create the simulation
release = "coast"  # 'coast' or 'off_shore'

fieldset.windage = 0.01  # windage factor for the particles

if release == "off_shore":
    lat0 = 52.10
    lon0 = 3.7
elif release == "coast":
    lat0 = 52.020000
    lon0 = 4.097500

radii = [100, 200, 400, 800, 1600]  # m
n_points = 16
R = 6371000  # Earth radius (m)

lat = [lat0]
lon = [lon0]
for r in radii:
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    dlat = np.rad2deg((r / R) * np.cos(theta))
    dlon = np.rad2deg((r / (R * np.cos(np.deg2rad(lat0)))) * np.sin(theta))
    lat.extend(lat0 + dlat)
    lon.extend(lon0 + dlon)

release_dt = np.timedelta64(12, "h")
nrepeat = np.timedelta64(28, "D") // release_dt  # number of releases
npart = len(lon)
lon = np.broadcast_to(lon, (nrepeat, npart))
lat = np.broadcast_to(lat, (nrepeat, npart))
time_i = fieldset.time_interval.left
time = (
    np.broadcast_to(time_i, (nrepeat, npart))
    + np.arange(0, nrepeat)[:, np.newaxis] * release_dt
)
print(
    f"Running {nrepeat} releases of {npart} particles each, for a total of {nrepeat * npart} particles."
)

MatroosParticle = parcels.Particle.add_variable(
    parcels.Variable("outside_stokes", dtype=np.int32, initial=0.0)
)
pset = parcels.ParticleSet(fieldset, pclass=MatroosParticle, x=lon, y=lat, t=time)

slurm_job_id = os.getenv("SLURM_JOB_ID", "local")
output_name = f"parcels-output-{slurm_job_id}.parquet"

outputdt = np.timedelta64(30, "m")  # output every 30 minutes
output_file = parcels.ParticleFile(
    output_name,
    outputdt=outputdt,
    mode="w",
)


def DeleteAnyError(particles, fieldset):
    any_error = particles.state >= 50  # This captures all Errors
    particles[any_error].state = parcels.StatusCode.Delete


@parcels.validate_kernel
def AdvectionRK2(particles, fieldset):  # pragma: no cover
    """Advection of particles using second-order Runge-Kutta integration."""
    (u1, v1) = fieldset.UV[particles]
    (us1, vs1) = fieldset.UVStokes[particles]
    (uw1, vw1) = fieldset.UVWind[particles]
    x1 = particles.x + (u1 + us1 + uw1 * fieldset.windage) * 0.5 * particles.dt
    y1 = particles.y + (v1 + vs1 + vw1 * fieldset.windage) * 0.5 * particles.dt

    (u2, v2) = fieldset.UV[
        particles.t + 0.5 * particles.dt, particles.z, y1, x1, particles
    ]
    (us2, vs2) = fieldset.UVStokes[
        particles.t + 0.5 * particles.dt, particles.z, y1, x1, particles
    ]
    (uw2, vw2) = fieldset.UVWind[
        particles.t + 0.5 * particles.dt, particles.z, y1, x1, particles
    ]

    # Handle particles that are outside the Stokes drift field
    outside_stokes = (us2 == 0) | (vs2 == 0)
    us2[outside_stokes] = 0.0
    vs2[outside_stokes] = 0.0
    particles.state[outside_stokes] = parcels.StatusCode.Success
    particles.outside_stokes[outside_stokes] = 1
    particles.outside_stokes[~outside_stokes] = 0

    # set wind to zero for particles that are on land
    on_land = (u2 == 0) & (v2 == 0)
    uw2[on_land] = 0.0
    vw2[on_land] = 0.0

    particles.dx += (u2 + us2 + uw2 * fieldset.windage) * particles.dt
    particles.dy += (v2 + vs2 + vw2 * fieldset.windage) * particles.dt


pset.execute(
    [AdvectionRK2, DeleteAnyError],
    endtime=fieldset.time_interval.right,
    dt=np.timedelta64(10, "m"),
    output_file=output_file,
)
