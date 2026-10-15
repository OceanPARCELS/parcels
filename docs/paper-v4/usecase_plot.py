import cmocean.cm as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import polars as pl
import xarray as xr

import parcels

DIR = "/Users/erik/Desktop/Parcelsv4-paper/"

# Load Parcels trajecories

df = parcels.read_particlefile(f"{DIR}/parcels-output-68694.parquet")

xlim = [df["x"].min(), df["x"].max()]
ylim = [df["y"].min(), df["y"].max()]

# Build a color map by release time (all particles released together share a color).
release_time_by_pid = df.group_by("particle_id").agg(
    pl.col("t").min().alias("release_time")
)
release_times = release_time_by_pid["release_time"].unique().sort().to_list()

colormap = mpl.colormaps["tab20b"]
release_to_color = {
    rt: colormap(i / max(len(release_times) - 1, 1))
    for i, rt in enumerate(release_times)
}
trajectory_to_color = {
    row["particle_id"]: release_to_color[row["release_time"]]
    for row in release_time_by_pid.iter_rows(named=True)
}

# Load flow, waves and wind data

ds_flow = xr.open_dataset(f"{DIR}/dcsm_fm100m_harmonie_202511010000.nc").isel(
    time=0, zc=0
)
zero_faces = (ds_flow["U"] == 0) & (ds_flow["V"] == 0)
ds_flow = ds_flow.isel(n_face=(~zero_faces).values)

triang = mtri.Triangulation(
    ds_flow["Mesh_node_x"].data,
    ds_flow["Mesh_node_y"].data,
    triangles=ds_flow["tri_face_nodes"].data,
)

ds_waves = xr.open_dataset(f"{DIR}/swan_kuststrook_harmonie_202511010000.nc").isel(
    time=0
)

ds_wind = xr.open_dataset(f"{DIR}/copernicusmarine_wind.nc").isel(time=0)


# Set up plotting canvas

speed_xlim = [1, 8]
speed_ylim = [50, 55]
speed_aspect = np.diff(speed_xlim)[0] / np.diff(speed_ylim)[0]
trajectory_aspect = np.diff(xlim)[0] / np.diff(ylim)[0]

figure_height = 8
figure_width = figure_height * (trajectory_aspect + speed_aspect / 3)
fig = plt.figure(figsize=(figure_width, figure_height), layout="constrained")
grid = fig.add_gridspec(
    3,
    2,
    width_ratios=[speed_aspect, 3 * trajectory_aspect],
)
speed_axes = [fig.add_subplot(grid[row, 0]) for row in range(3)]
particle_ax = fig.add_subplot(grid[:, 1])

# Plot particle trajectories and source meshes

particle_ax.set_xlim(xlim)
particle_ax.set_ylim(ylim)

for column in range(ds_waves["lon"].values.shape[1]):
    label = "Waves mesh" if column == 0 else None
    particle_ax.plot(
        ds_waves["lon"].values[:, column],
        ds_waves["lat"].values[:, column],
        color="b",
        lw=0.3,
        alpha=0.5,
        label=label,
    )
for row in range(ds_waves["lon"].values.shape[0]):
    particle_ax.plot(
        ds_waves["lon"].values[row, :],
        ds_waves["lat"].values[row, :],
        color="b",
        lw=0.3,
        alpha=0.5,
    )

for latitude_index in range(ds_wind["latitude"].values.shape[0]):
    label = "Wind mesh" if latitude_index == 0 else None
    particle_ax.hlines(
        ds_wind["latitude"].values[latitude_index],
        xmin=ds_wind["longitude"].values.min(),
        xmax=ds_wind["longitude"].values.max(),
        color="r",
        lw=0.3,
        alpha=0.5,
        label=label,
        zorder=0,
    )
for longitude_index in range(ds_wind["longitude"].values.shape[0]):
    particle_ax.vlines(
        ds_wind["longitude"].values[longitude_index],
        ymin=ds_wind["latitude"].values.min(),
        ymax=ds_wind["latitude"].values.max(),
        color="r",
        lw=0.3,
        alpha=0.5,
        zorder=1,
    )

particle_ax.triplot(
    triang,
    color="k",
    lw=0.3,
    alpha=0.5,
    label="Flow mesh",
    zorder=2,
)

for particle_id in df["particle_id"].unique():
    trajectory = df.filter(pl.col("particle_id") == particle_id)
    particle_ax.plot(
        trajectory["x"],
        trajectory["y"],
        color=trajectory_to_color[particle_id],
        linewidth=0.6,
        alpha=0.3,
    )
    particle_ax.plot(
        trajectory["x"][-1],
        trajectory["y"][-1],
        marker="o",
        color=trajectory_to_color[particle_id],
        markersize=3,
        zorder=4,
    )

legend_handles, legend_labels = particle_ax.get_legend_handles_labels()
particle_ax.legend(
    legend_handles[::-1],
    legend_labels[::-1],
    loc="lower right",
)
legend = particle_ax.get_legend()
for line in legend.get_lines():
    line.set_linewidth(2)

particle_ax.set_aspect("equal", adjustable="box")
particle_ax.set_title("Parcels trajectories")
particle_ax.set_xlabel("Longitude [°E]")
particle_ax.set_ylabel("Latitude [°N]")


# Speed maps
clim = [0, 0.5]
label_box = {
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.8,
    "boxstyle": "round,pad=0.3",
}

# Flow speed
flowspeed = np.hypot(ds_flow["U"], ds_flow["V"])
flow_map = speed_axes[0].tripcolor(
    triang,
    facecolors=flowspeed.to_numpy(),
    cmap=cmo.speed,
    shading="flat",
    clim=clim,
)

# Wave speed
Us = (
    2
    * np.pi**3
    * ds_waves["wave_height_hm0"] ** 2
    / (ds_waves["wave_period_tm10"] ** 3 * 9.81)
)
ds_waves["Us"] = Us * np.cos(ds_waves["wave_dir_th0"] * np.pi / 180)
ds_waves["Vs"] = Us * np.sin(ds_waves["wave_dir_th0"] * np.pi / 180)
wavespeed = np.hypot(ds_waves["Us"], ds_waves["Vs"])
wave_map = speed_axes[1].pcolor(
    ds_waves["lon"],
    ds_waves["lat"],
    wavespeed,
    cmap=cmo.speed,
    clim=clim,
)

# Wind speed
windspeed = np.hypot(ds_wind["eastward_wind"], ds_wind["northward_wind"])
wind_map = speed_axes[2].pcolor(
    ds_wind["longitude"],
    ds_wind["latitude"],
    windspeed * 0.01,
    cmap=cmo.speed,
    clim=clim,
)

for speed_ax, label in zip(speed_axes, ["Flow", "Waves", "1% of wind"], strict=True):
    speed_ax.text(
        0.98,
        0.04,
        label,
        transform=speed_ax.transAxes,
        ha="right",
        va="bottom",
        bbox=label_box,
    )
    speed_ax.set_xlim(speed_xlim)
    speed_ax.set_ylim(speed_ylim)
    speed_ax.set_aspect("equal", adjustable="box")

colorbar_ax = speed_axes[2].inset_axes(
    [0, -0.34, 1, 0.1],
    transform=speed_axes[2].transAxes,
    zorder=5,
)
fig.colorbar(
    wind_map,
    cax=colorbar_ax,
    orientation="horizontal",
    label="Speed [m s$^{-1}$]",
)
plt.savefig("usecase_plot.png", dpi=300, bbox_inches="tight")
