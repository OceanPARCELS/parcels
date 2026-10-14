"""Datasets mimicking the layout of real-world hydrodynamic models"""

import numpy as np
import xarray as xr

from . import T, X, Y, Z

__all__ = ["T", "X", "Y", "Z", "datasets"]

TIME = np.datetime64("2000-01-01") + np.arange(T) * np.timedelta64(1, "D")


def _copernicusmarine():
    """Copernicus Marine Service dataset as retrieved by the `copernicusmarine` toolkit"""
    return xr.Dataset(
        {
            "uo": (
                ["time", "depth", "latitude", "longitude"],
                np.random.rand(T, Z, Y, X),
                {
                    "valid_max": 5.0,
                    "unit_long": "Meters per second",
                    "units": "m s-1",
                    "long_name": "Eastward velocity",
                    "standard_name": "eastward_sea_water_velocity",
                    "valid_min": -5.0,
                },
            ),
            "vo": (
                ["time", "depth", "latitude", "longitude"],
                np.random.rand(T, Z, Y, X),
                {
                    "valid_max": 5.0,
                    "unit_long": "Meters per second",
                    "units": "m s-1",
                    "long_name": "Northward velocity",
                    "standard_name": "northward_sea_water_velocity",
                    "valid_min": -5.0,
                },
            ),
        },
        coords={
            "depth": (
                ["depth"],
                np.linspace(0.49, 5727.92, Z),
                {
                    "unit_long": "Meters",
                    "units": "m",
                    "axis": "Z",
                    "long_name": "Depth",
                    "standard_name": "depth",
                    "positive": "down",
                },
            ),
            "latitude": (
                ["latitude"],
                np.linspace(-90, 90, Y),
                {
                    "unit_long": "Degrees North",
                    "units": "degrees_north",
                    "axis": "Y",
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                },
            ),
            "longitude": (
                ["longitude"],
                np.linspace(-180, 180, X),
                {
                    "unit_long": "Degrees East",
                    "units": "degrees_east",
                    "axis": "X",
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                },
            ),
            "time": (
                ["time"],
                TIME,
                {
                    "unit_long": "Hours Since 1950-01-01",
                    "axis": "T",
                    "long_name": "Time",
                    "standard_name": "time",
                },
            ),
        },
    )


def _copernicusmarine_waves():
    """Copernicus Marine Service GlobCurrent dataset (MULTIOBS_GLO_PHY_MYNRT_015_003)"""
    return xr.Dataset(
        {
            "VSDX": (
                ["time", "depth", "latitude", "longitude"],
                np.random.rand(T, Z, Y, X),
                {
                    "units": "m s-1",
                    "standard_name": "sea_surface_wave_stokes_drift_x_velocity",
                    "long_name": "Stokes drift U",
                    "WMO_code": 215,
                    "cell_methods": "time:point area:mean",
                    "missing_value": -32767,
                    "type_of_analysis": "spectral analysis",
                },
            ),
            "VSDY": (
                ["time", "depth", "latitude", "longitude"],
                np.random.rand(T, Z, Y, X),
                {
                    "units": "m s-1",
                    "standard_name": "sea_surface_wave_stokes_drift_y_velocity",
                    "long_name": "Stokes drift V",
                    "WMO_code": 216,
                    "cell_methods": "time:point area:mean",
                    "missing_value": -32767,
                    "type_of_analysis": "spectral analysis",
                },
            ),
        },
        coords={
            "depth": (
                ["depth"],
                np.linspace(-0.0, 15, Z),
                {
                    "standard_name": "depth",
                    "long_name": "Depth",
                    "units": "m",
                    "unit_long": "Meters",
                    "axis": "Z",
                    "positive": "down",
                },
            ),
            "latitude": (
                ["latitude"],
                np.linspace(-90, 90, Y),
                {
                    "unit_long": "Degrees North",
                    "units": "degrees_north",
                    "axis": "Y",
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                },
            ),
            "longitude": (
                ["longitude"],
                np.linspace(-180, 180, X),
                {
                    "unit_long": "Degrees East",
                    "units": "degrees_east",
                    "axis": "X",
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                },
            ),
            "time": (
                ["time"],
                TIME,
                {
                    "axis": "T",
                    "long_name": "Time",
                    "standard_name": "time",
                },
            ),
        },
    )


def _NEMO_MOI_U():
    """NEMO model dataset (U component) as serviced by Mercator Ocean International"""
    return xr.Dataset(
        {
            "vozocrtx": (
                ["deptht", "y", "x"],
                np.random.rand(Z, Y, X),
                {
                    "units": "m s-1",
                    "valid_min": -10.0,
                    "valid_max": 10.0,
                    "long_name": "Zonal velocity",
                    "standard_name": "sea_water_x_velocity",
                    "short_name": "vozocrtx",
                    "online_operation": "N/A",
                    "interval_operation": 86400,
                    "interval_write": 86400,
                    "associate": "time_counter deptht nav_lat nav_lon",
                },
            ),
            "sotkeavmu1": (
                ["y", "x"],
                np.random.rand(Y, X),
                {
                    "units": "m2 s-1",
                    "valid_min": 0.0,
                    "valid_max": 100.0,
                    "long_name": "Vertical Eddy Viscosity U 1m",
                    "standard_name": "ocean_vertical_eddy_viscosity_u_1m",
                    "short_name": "sotkeavmu1",
                    "online_operation": "N/A",
                    "interval_operation": 86400,
                    "interval_write": 86400,
                    "associate": "time_counter nav_lat nav_lon",
                },
            ),
        },
        coords={
            "nav_lon": (
                ["y", "x"],
                np.tile(np.linspace(-179, 179, X, endpoint=False), (Y, 1)),  # note that this is not curvilinear
                {
                    "units": "degrees_east",
                    "valid_min": -179.99984754002182,
                    "valid_max": 179.999842386314,
                    "long_name": "Longitude",
                    "nav_model": "Default grid",
                    "standard_name": "longitude",
                },
            ),
            "nav_lat": (
                ["y", "x"],
                np.tile(np.linspace(-75, 85, Y).reshape(-1, 1), (1, X)),  # note that this is not curvilinear
                {
                    "units": "degrees_north",
                    "valid_min": -77.0104751586914,
                    "valid_max": 89.9591064453125,
                    "long_name": "Latitude",
                    "nav_model": "Default grid",
                    "standard_name": "latitude",
                },
            ),
            "x": (
                ["x"],
                np.arange(X, dtype="int32"),
                {
                    "standard_name": "projection_x_coordinate",
                    "axis": "X",
                    "units": "1",
                },
            ),
            "y": (
                ["y"],
                np.arange(Y, dtype="int32"),
                {
                    "standard_name": "projection_y_coordinate",
                    "axis": "Y",
                    "units": "1",
                },
            ),
            "deptht": (
                ["deptht"],
                np.linspace(1, 5500, Z, dtype="float64"),
                {
                    "units": "m",
                    "positive": "down",
                    "valid_min": 0.4940253794193268,
                    "valid_max": 5727.91650390625,
                    "long_name": "Vertical T levels",
                    "standard_name": "depth",
                    "axis": "Z",
                },
            ),
        },
    )


def _NEMO_MOI_V():
    """NEMO model dataset (V component) as serviced by Mercator Ocean International"""
    return xr.Dataset(
        {
            "vomecrty": (
                ["deptht", "y", "x"],
                np.random.rand(Z, Y, X),
                {
                    "units": "m s-1",
                    "valid_min": -10.0,
                    "valid_max": 10.0,
                    "long_name": "Meridional velocity",
                    "standard_name": "sea_water_y_velocity",
                    "short_name": "vomecrty",
                    "online_operation": "N/A",
                    "interval_operation": 86400,
                    "interval_write": 86400,
                    "associate": "time_counter deptht nav_lat nav_lon",
                },
            ),
        },
        coords={
            "nav_lon": (
                ["y", "x"],
                np.tile(np.linspace(-179, 179, X, endpoint=False), (Y, 1)),  # note that this is not curvilinear
                {
                    "units": "degrees_east",
                    "valid_min": -179.9999951021171,
                    "valid_max": 180.0,
                    "long_name": "Longitude",
                    "nav_model": "Default grid",
                    "standard_name": "longitude",
                },
            ),
            "nav_lat": (
                ["y", "x"],
                np.tile(np.linspace(-75, 85, Y).reshape(-1, 1), (1, X)),  # note that this is not curvilinear
                {
                    "units": "degrees_north",
                    "valid_min": -77.00110752801133,
                    "valid_max": 89.95529158641207,
                    "long_name": "Latitude",
                    "nav_model": "Default grid",
                    "standard_name": "latitude",
                },
            ),
            "x": (
                ["x"],
                np.arange(X, dtype="int32"),
                {
                    "standard_name": "projection_x_coordinate",
                    "axis": "X",
                    "units": "1",
                },
            ),
            "y": (
                ["y"],
                np.arange(Y, dtype="int32"),
                {
                    "standard_name": "projection_y_coordinate",
                    "axis": "Y",
                    "units": "1",
                },
            ),
            "deptht": (
                ["deptht"],
                np.linspace(1, 5500, Z, dtype="float64"),
                {
                    "units": "m",
                    "positive": "down",
                    "valid_min": 0.4940253794193268,
                    "valid_max": 5727.91650390625,
                    "long_name": "Vertical T levels",
                    "standard_name": "depth",
                    "axis": "Z",
                },
            ),
        },
    )


def _CESM():
    """CESM model dataset"""
    return xr.Dataset(
        {
            "UVEL": (
                ["time", "z_t", "nlat", "nlon"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "long_name": "Velocity in grid-x direction",
                    "units": "centimeter/s",
                    "grid_loc": "3221",
                    "cell_methods": "time: mean",
                },
            ),
            "VVEL": (
                ["time", "z_t", "nlat", "nlon"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "long_name": "Velocity in grid-y direction",
                    "units": "centimeter/s",
                    "grid_loc": "3221",
                    "cell_methods": "time: mean",
                },
            ),
            "WVEL": (
                ["time", "z_w_top", "nlat", "nlon"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "long_name": "Vertical Velocity",
                    "units": "centimeter/s",
                    "grid_loc": "3112",
                    "cell_methods": "time: mean",
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                np.linspace(0, 5000, T),
                {
                    "long_name": "time",
                    "bounds": "time_bound",
                },
            ),
            "z_t": (
                ["z_t"],
                np.linspace(0, 5000, Z, dtype="float32"),
                {
                    "long_name": "depth from surface to midpoint of layer",
                    "units": "centimeters",
                    "positive": "down",
                    "valid_min": 500.0,
                    "valid_max": 537500.0,
                },
            ),
            "z_w_top": (
                ["z_w_top"],
                np.linspace(0, 5000, Z, dtype="float32"),
                {
                    "long_name": "depth from surface to top of layer",
                    "units": "centimeters",
                    "positive": "down",
                    "valid_min": 0.0,
                    "valid_max": 525000.9375,
                },
            ),
            "ULONG": (
                ["nlat", "nlon"],
                np.tile(np.linspace(-179, 179, X, endpoint=False), (Y, 1)),  # note that this is not curvilinear
                {
                    "long_name": "array of u-grid longitudes",
                    "units": "degrees_east",
                },
            ),
            "ULAT": (
                ["nlat", "nlon"],
                np.tile(np.linspace(-75, 85, Y).reshape(-1, 1), (1, X)),  # note that this is not curvilinear
                {
                    "long_name": "array of u-grid latitudes",
                    "units": "degrees_north",
                },
            ),
        },
    )


def _MITgcm_netcdf():
    """MITgcm model dataset in netCDF format"""
    return xr.Dataset(
        #
        {
            "U": (
                ["T", "Z", "Y", "Xp1"],
                np.random.rand(T, Z, Y, X + 1).astype("float32"),
                {
                    "units": "m/s",
                    "coordinates": "XU YU RC iter",
                },
            ),
            "V": (
                ["T", "Z", "Yp1", "X"],
                np.random.rand(T, Z, Y + 1, X).astype("float32"),
                {
                    "units": "m/s",
                    "coordinates": "XV YV RC iter",
                },
            ),
            "W": (
                ["T", "Zl", "Y", "X"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "units": "m/s",
                    "coordinates": "XC YC RC iter",
                },
            ),
            "Temp": (
                ["T", "Z", "Y", "X"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "units": "degC",
                    "coordinates": "XC YC RC iter",
                    "long_name": "potential_temperature",
                },
            ),
        },
        coords={
            "T": (
                ["T"],
                np.arange(0, T, dtype="float64"),
                {
                    "long_name": "model_time",
                    "units": "s",
                },
            ),
            "Z": (
                ["Z"],
                np.linspace(-25, -5000, Z, dtype="float64"),
                {
                    "long_name": "vertical coordinate of cell center",
                    "units": "meters",
                    "positive": "up",
                },
            ),
            "Zl": (
                ["Zl"],
                np.linspace(0, -4500, Z, dtype="float64"),
                {
                    "long_name": "vertical coordinate of upper cell interface",
                    "units": "meters",
                    "positive": "up",
                },
            ),
            "Y": (
                ["Y"],
                np.linspace(500, 5000, Y, dtype="float64"),
                {
                    "long_name": "Y-Coordinate of cell center",
                    "units": "meters",
                },
            ),
            "Yp1": (
                ["Yp1"],
                np.linspace(0, 4500, Y + 1, dtype="float64"),
                {
                    "long_name": "Y-Coordinate of cell corner",
                    "units": "meters",
                },
            ),
            "X": (
                ["X"],
                np.linspace(500, 5000, X, dtype="float64"),
                {
                    "long_name": "X-coordinate of cell center",
                    "units": "meters",
                },
            ),
            "Xp1": (
                ["Xp1"],
                np.linspace(0, 4100, X + 1, dtype="float64"),
                {
                    "long_name": "X-Coordinate of cell corner",
                    "units": "meters",
                },
            ),
        },
    )


def _MITgcm_mds():
    """MITgcm model dataset in native MDS format"""
    return xr.Dataset(
        {
            "U": (
                ["time", "Z", "YC", "XG"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "standard_name": "sea_water_x_velocity",
                    "mate": "V",
                    "long_name": "Zonal Component of Velocity",
                    "units": "m s-1",
                },
            ),
            "V": (
                ["time", "Z", "YG", "XC"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "standard_name": "sea_water_y_velocity",
                    "mate": "U",
                    "long_name": "Meridional Component of Velocity",
                    "units": "m s-1",
                },
            ),
            "W": (
                ["time", "Zl", "YC", "XC"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "standard_name": "sea_water_z_velocity",
                    "long_name": "Vertical Component of Velocity",
                    "units": "m s-1",
                },
            ),
            "S": (
                ["time", "Z", "YC", "XC"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "standard_name": "sea_water_salinity",
                    "long_name": "Salinity",
                    "units": "g kg-1",
                },
            ),
            "T": (
                ["time", "Z", "YC", "XC"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "standard_name": "sea_water_potential_temperature",
                    "long_name": "Potential Temperature",
                    "units": "degree_Celcius",
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                np.arange(T) * np.timedelta64(1, "D"),
                {
                    "standard_name": "time",
                    "long_name": "Time",
                    "axis": "T",
                    "calendar": "gregorian",
                },
            ),
            "Z": (
                ["Z"],
                np.linspace(-25, -5000, Z, dtype="float64"),
                {
                    "standard_name": "depth",
                    "long_name": "vertical coordinate of cell center",
                    "units": "m",
                    "positive": "down",
                    "axis": "Z",
                },
            ),
            "Zl": (
                ["Zl"],
                np.linspace(0, -4500, Z, dtype="float64"),
                {
                    "standard_name": "depth_at_lower_w_location",
                    "long_name": "vertical coordinate of lower cell interface",
                    "units": "m",
                    "positive": "down",
                    "axis": "Z",
                },
            ),
            "YC": (
                ["YC"],
                np.linspace(500, 5000, Y, dtype="float64"),
                {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "coordinate": "YC XC",
                    "axis": "Y",
                },
            ),
            "YG": (
                ["YG"],
                np.linspace(0, 5000, Y, dtype="float64"),
                {
                    "standard_name": "latitude_at_f_location",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "coordinate": "YG XG",
                    "axis": "Y",
                },
            ),
            "XC": (
                ["XC"],
                np.linspace(500, 5000, X, dtype="float64"),
                {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "coordinate": "YC XC",
                    "axis": "X",
                },
            ),
            "XG": (
                ["XG"],
                np.linspace(0, 5000, X, dtype="float64"),
                {
                    "standard_name": "longitude_at_f_location",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "coordinate": "YG XG",
                    "axis": "X",
                },
            ),
        },
    )


def _ERA5_wind():
    """ERA5 10m wind model dataset"""
    return xr.Dataset(
        {
            "u10": (
                ["time", "latitude", "longitude"],
                np.random.rand(T, Y, X).astype("float32"),
                {
                    "long_name": "10 metre U wind component",
                    "units": "m s**-1",
                },
            ),
            "v10": (
                ["time", "latitude", "longitude"],
                np.random.rand(T, Y, X).astype("float32"),
                {
                    "long_name": "10 metre V wind component",
                    "units": "m s**-1",
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                TIME,
                {
                    "long_name": "time",
                },
            ),
            "latitude": (
                ["latitude"],
                np.linspace(90, -90, Y),  # Note: ERA5 uses latitudes from 90 to -90
                {
                    "long_name": "latitude",
                    "units": "degrees_north",
                },
            ),
            "longitude": (
                ["longitude"],
                np.linspace(0, 360, X, endpoint=False),
                {
                    "long_name": "longitude",
                    "units": "degrees_east",
                },
            ),
        },
    )


def _FES_tides():
    """FES tidal model dataset"""
    return xr.Dataset(
        {
            "Ug": (
                ["lat", "lon"],
                np.random.rand(Y, X).astype("float32"),
                {
                    "long_name": "Eastward sea water velocity phaselag due to non equilibrium ocean tide at m2 frequency",
                    "units": "degrees",
                    "grid_mapping": "crs",
                },
            ),
            "Ua": (
                ["lat", "lon"],
                np.random.rand(Y, X).astype("float32"),
                {
                    "long_name": "Eastward sea water velocity amplitude due to non equilibrium ocean tide at m2 frequency",
                    "units": "cm/s",
                    "grid_mapping": "crs",
                },
            ),
        },
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, Y),
                {
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "bounds": "lat_bnds",
                    "axis": "Y",
                    "valid_min": -90.0,
                    "valid_max": 90.0,
                },
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, X, endpoint=False),
                {
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "bounds": "lon_bnds",
                    "axis": "X",
                    "valid_min": 0.0,
                    "valid_max": 360.0,
                },
            ),
        },
    )


def _hycom_espc():
    """HYCOM ESPC model dataset from https://data.hycom.org/datasets/ESPC-D-V02/data/daily_netcdf/2025/"""
    return xr.Dataset(
        {
            "water_u": (
                ["time", "depth", "lat", "lon"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "long_name": "Eastward Water Velocity",
                    "standard_name": "eastward_sea_water_velocity",
                    "units": "m/s",
                    "NAVO_code": 17,
                    "actual_range": np.array([-3.3700001, 3.6840003], dtype="float32"),
                    "cell_methods": "time: mean",
                },
            ),
            "tau": (
                ["time"],
                np.linspace(0, 24, T, dtype="float64"),
                {
                    "long_name": "Tau",
                    "units": "hours since analysis",
                    "time_origin": "2024-12-31 12:00:00",
                    "NAVO_code": 56,
                    "cell_methods": "time: mean",
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                np.arange(0, T, dtype="float64"),
                {
                    "long_name": "Valid Time",
                    "units": "hours since 2000-01-01 00:00:00",
                    "time_origin": "2000-01-01 00:00:00",
                    "calendar": "standard",
                    "axis": "T",
                    "NAVO_code": 13,
                    "cell_methods": "time: mean",
                },
            ),
            "depth": (
                ["depth"],
                np.linspace(0, 5000, Z, dtype="float32"),
                {
                    "long_name": "Depth",
                    "standard_name": "depth",
                    "units": "m",
                    "positive": "down",
                    "axis": "Z",
                    "NAVO_code": 5,
                },
            ),
            "lat": (
                ["lat"],
                np.linspace(-80, 90, Y),
                {
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                    "point_spacing": "even",
                    "axis": "Y",
                    "NAVO_code": 1,
                },
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, X, endpoint=False),
                {
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                    "modulo": "360 degrees",
                    "axis": "X",
                    "NAVO_code": 2,
                },
            ),
        },
    )


def _ecco4():
    """ECCO V4r4 model dataset (from https://podaac.jpl.nasa.gov/dataset/ECCO_L4_OCEAN_VEL_LLC0090GRID_DAILY_V4R4#capability-modal-download)"""
    tiles = 13
    lon_grid = np.tile(
        np.tile(np.linspace(-179, 179, X, endpoint=False), (Y, 1)), (tiles, 1, 1)
    )  # NOTE this grid is not correct, as duplicates for each tile
    lat_grid = np.tile(
        np.tile(np.linspace(-89, 89, Y), (X, 1)).T, (tiles, 1, 1)
    )  # NOTE this grid is not correct, as duplicates for each tile
    return xr.Dataset(
        {
            "UVEL": (
                ["time", "k", "tile", "j", "i_g"],
                np.random.rand(T, Z, tiles, Y, X).astype("float32"),
                {
                    "long_name": "Horizontal velocity in the model +x direction",
                    "units": "m s-1",
                    "mate": "VVEL",
                    "coverage_content_type": "modelResult",
                    "direction": ">0 increases volume",
                    "standard_name": "sea_water_x_velocity",
                    "comment": "Horizontal velocity in the +x direction at the 'u' face of the tracer cell on the native model grid. Note: in the Arakawa-C grid, horizontal velocities are staggered relative to the tracer cells with indexing such that +UVEL(i_g,j,k) corresponds to +x fluxes through the 'u' face of the tracer cell at (i,j,k). Do NOT use UVEL for volume flux calculations because the model's grid cell thicknesses vary with time (z* coordinates); use UVELMASS instead. Also, the model +x direction does not necessarily correspond to the geographical east-west direction because the x and y axes of the model's curvilinear lat-lon-cap (llc) grid have arbitrary orientations which vary within and across tiles. See EVEL and NVEL for zonal and meridional velocity.",
                    "valid_min": -2.139253616333008,
                    "valid_max": 2.038635015487671,
                },
            ),
            "VVEL": (
                ["time", "k", "tile", "j_g", "i"],
                np.random.rand(T, Z, tiles, Y, X).astype("float32"),
                {
                    "long_name": "Horizontal velocity in the model +y direction",
                    "units": "m s-1",
                    "mate": "UVEL",
                    "coverage_content_type": "modelResult",
                    "direction": ">0 increases volume",
                    "standard_name": "sea_water_y_velocity",
                    "comment": "Horizontal velocity in the +y direction at the 'v' face of the tracer cell on the native model grid. Note: in the Arakawa-C grid, horizontal velocities are staggered relative to the tracer cells with indexing such that +VVEL(i,j_g,k) corresponds to +y fluxes through the 'v' face of the tracer cell at (i,j,k). Do NOT use VVEL for volume flux calculations because the model's grid cell thicknesses vary with time (z* coordinates); use VVELMASS instead. Also, the model +y direction does not necessarily correspond to the geographical north-south direction because the x and y axes of the model's curvilinear lat-lon-cap (llc) grid have arbitrary orientations which vary within and across tiles. See EVEL and NVEL for zonal and meridional velocity.",
                    "valid_min": -1.7877743244171143,
                    "valid_max": 1.9089667797088623,
                },
            ),
            "WVEL": (
                ["time", "k_l", "tile", "j", "i"],
                np.random.rand(T, Z, tiles, Y, X).astype("float32"),
                {
                    "long_name": "Vertical velocity",
                    "units": "m s-1",
                    "coverage_content_type": "modelResult",
                    "direction": ">0 decreases volume",
                    "standard_name": "upward_sea_water_velocity",
                    "comment": "Vertical velocity in the +z direction at the top 'w' face of the tracer cell on the native model grid. Note: in the Arakawa-C grid, vertical velocities are staggered relative to the tracer cells with indexing such that +WVEL(i,j,k_l) corresponds to upward +z motion through the top 'w' face of the tracer cell at (i,j,k). WVEL is identical to WVELMASS.",
                    "valid_min": -0.0023150660563260317,
                    "valid_max": 0.0016380994347855449,
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                TIME,
                {
                    "long_name": "center time of averaging period",
                    "standard_name": "time",
                    "axis": "T",
                    "bounds": "time_bnds",
                    "coverage_content_type": "coordinate",
                },
            ),
            "tile": (
                ["tile"],
                np.arange(tiles, dtype="int32"),
                {
                    "long_name": "lat-lon-cap tile index",
                    "coverage_content_type": "coordinate",
                    "comment": "The ECCO V4 horizontal model grid is divided into 13 tiles of 90x90 cells for convenience.",
                },
            ),
            "k": (
                ["k"],
                np.arange(Z, dtype="int32"),
                {
                    "long_name": "grid index in z for tracer variables",
                    "axis": "Z",
                    "swap_dim": "Z",
                    "coverage_content_type": "coordinate",
                },
            ),
            "k_l": (
                ["k_l"],
                np.arange(Z, dtype="int32"),
                {
                    "long_name": "grid index in z corresponding to the top face of tracer grid cells ('w' locations)",
                    "axis": "Z",
                    "swap_dim": "Zl",
                    "coverage_content_type": "coordinate",
                    "c_grid_axis_shift": -0.5,
                    "comment": "First index corresponds to the top surface of the uppermost tracer grid cell. The use of 'l' in the variable name follows the MITgcm convention for ocean variables in which the lower (l) face of a tracer grid cell on the logical grid corresponds to the top face of the grid cell on the physical grid.",
                },
            ),
            "j": (
                ["j"],
                np.arange(Y, dtype="int32"),
                {
                    "long_name": "grid index in y for variables at tracer and 'u' locations",
                    "axis": "Y",
                    "swap_dim": "YC",
                    "coverage_content_type": "coordinate",
                    "comment": "In the Arakawa C-grid system, tracer (e.g., THETA) and 'u' variables (e.g., UVEL) have the same y coordinate on the model grid.",
                },
            ),
            "j_g": (
                ["j_g"],
                np.arange(Y, dtype="int32"),
                {
                    "long_name": "grid index in y for variables at 'v' and 'g' locations",
                    "axis": "Y",
                    "swap_dim": "YG",
                    "c_grid_axis_shift": -0.5,
                    "coverage_content_type": "coordinate",
                    "comment": "In the Arakawa C-grid system, 'v' (e.g., VVEL) and 'g' variables (e.g., XG) have the same y coordinate.",
                },
            ),
            "i": (
                ["i"],
                np.arange(X, dtype="int32"),
                {
                    "long_name": "grid index in x for variables at tracer and 'v' locations",
                    "axis": "X",
                    "swap_dim": "XC",
                    "coverage_content_type": "coordinate",
                    "comment": "In the Arakawa C-grid system, tracer (e.g., THETA) and 'v' variables (e.g., VVEL) have the same x coordinate on the model grid.",
                },
            ),
            "i_g": (
                ["i_g"],
                np.arange(X, dtype="int32"),
                {
                    "long_name": "grid index in x for variables at 'u' and 'g' locations",
                    "axis": "X",
                    "swap_dim": "XG",
                    "c_grid_axis_shift": -0.5,
                    "coverage_content_type": "coordinate",
                    "comment": "In the Arakawa C-grid system, 'u' (e.g., UVEL) and 'g' variables (e.g., XG) have the same x coordinate on the model grid.",
                },
            ),
            "Z": (
                ["k"],
                np.linspace(-5, -5900, Z, dtype="float32"),
                {
                    "long_name": "depth of tracer grid cell center",
                    "standard_name": "depth",
                    "units": "m",
                    "positive": "up",
                    "bounds": "Z_bnds",
                    "coverage_content_type": "coordinate",
                    "comment": "Non-uniform vertical spacing.",
                },
            ),
            "Zl": (
                ["k_l"],
                np.linspace(0, -5678, Z, dtype="float32"),
                {
                    "long_name": "depth of the top face of tracer grid cells",
                    "standard_name": "depth",
                    "units": "m",
                    "positive": "up",
                    "coverage_content_type": "coordinate",
                    "comment": "First element is 0m, the depth of the top face of the first tracer grid cell (ocean surface). Last element is the depth of the top face of the deepest grid cell. The use of 'l' in the variable name follows the MITgcm convention for ocean variables in which the lower (l) face of a tracer grid cell on the logical grid corresponds to the top face of the grid cell on the physical grid. In other words, the logical vertical grid of MITgcm ocean variables is inverted relative to the physical vertical grid.",
                },
            ),
            "YC": (
                ["tile", "j", "i"],
                lat_grid,
                {
                    "long_name": "latitude of tracer grid cell center",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                    "coordinate": "YC XC",
                    "bounds": "YC_bnds",
                    "coverage_content_type": "coordinate",
                    "comment": "nonuniform grid spacing",
                },
            ),
            "YG": (
                ["tile", "j_g", "i_g"],
                lat_grid,
                {
                    "long_name": "latitude of 'southwest' corner of tracer grid cell",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                    "coordinate": "YG XG",
                    "coverage_content_type": "coordinate",
                    "comment": "Nonuniform grid spacing. Note: 'southwest' does not correspond to geographic orientation but is used for convenience to describe the computational grid. See MITgcm dcoumentation for details.",
                },
            ),
            "XC": (
                ["tile", "j", "i"],
                lon_grid,
                {
                    "long_name": "longitude of tracer grid cell center",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                    "coordinate": "YC XC",
                    "bounds": "XC_bnds",
                    "coverage_content_type": "coordinate",
                    "comment": "nonuniform grid spacing",
                },
            ),
            "XG": (
                ["tile", "j_g", "i_g"],
                lon_grid,
                {
                    "long_name": "longitude of 'southwest' corner of tracer grid cell",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                    "coordinate": "YG XG",
                    "coverage_content_type": "coordinate",
                    "comment": "Nonuniform grid spacing. Note: 'southwest' does not correspond to geographic orientation but is used for convenience to describe the computational grid. See MITgcm dcoumentation for details.",
                },
            ),
        },
    )


def _CROCO_idealized():
    """CROCO idealized model dataset"""
    return xr.Dataset(
        {
            "u": (
                ["time", "s_rho", "eta_rho", "xi_u"],
                np.random.rand(T, Z, Y, X - 1).astype("float32"),
                {
                    "long_name": "u-momentum component",
                    "units": "meter second-1",
                    "field": "u-velocity, scalar, series",
                    "standard_name": "sea_water_x_velocity_at_u_location",
                },
            ),
            "v": (
                ["time", "s_rho", "eta_v", "xi_rho"],
                np.random.rand(T, Z, Y - 1, X).astype("float32"),
                {
                    "long_name": "v-momentum component",
                    "units": "meter second-1",
                    "field": "v-velocity, scalar, series",
                    "standard_name": "sea_water_y_velocity_at_v_location",
                },
            ),
            "w": (
                ["time", "s_rho", "eta_rho", "xi_rho"],
                np.random.rand(T, Z, Y, X).astype("float32"),
                {
                    "long_name": "vertical momentum component",
                    "units": "meter second-1",
                    "field": "w-velocity, scalar, series",
                    "standard_name": "upward_sea_water_velocity",
                    "coordinates": "lat_rho lon_rho",
                },
            ),
            "h": (
                ["eta_rho", "xi_rho"],
                np.random.rand(Y, X).astype("float32"),
                {
                    "long_name": "bathymetry at RHO-points",
                    "units": "meter",
                    "field": "bath, scalar",
                    "standard_name": "model_sea_floor_depth_below_geoid",
                },
            ),
            "zeta": (
                ["time", "eta_rho", "xi_rho"],
                np.random.rand(T, Y, X).astype("float32"),
                {
                    "long_name": "free-surface",
                    "units": "meter",
                    "field": "free-surface, scalar, series",
                    "standard_name": "sea_surface_height",
                },
            ),
            "Cs_w": (
                ["s_w"],
                np.random.rand(Z + 1).astype("float32"),
                {
                    "long_name": "S-coordinate stretching curves at W-points",
                },
            ),
            "hc": (
                [],
                np.array(0.0, dtype="float32"),
                {
                    "long_name": "S-coordinate parameter, critical depth",
                    "units": "meter",
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                np.arange(0, T, dtype="float64"),
                {
                    "long_name": "time since initialization",
                    "units": "second",
                    "field": "time, scalar, series",
                    "standard_name": "time",
                    "axis": "T",
                },
            ),
            "s_rho": (
                ["s_rho"],
                np.linspace(-0.95, 0.05, Z, dtype="float32"),
                {
                    "long_name": "S-coordinate at RHO-points",
                    "standard_name": "ocean_s_coordinate_g1",
                    "positive": "up",
                    "axis": "Z",
                    "formula_terms": "s: sc_r C: Cs_r eta: zeta depth: h depth_c: hc",
                },
            ),
            "s_w": (
                ["s_w"],
                np.linspace(-1, 0, Z + 1, dtype="float32"),
                {
                    "long_name": "S-coordinate at W-points",
                    "standard_name": "ocean_s_coordinate_g1_at_w_location",
                    "positive": "up",
                    "axis": "Z",
                    "c_grid_axis_shift": -0.5,
                    "formula_terms": "s: sc_w C: Cs_w eta: zeta depth: h depth_c: hc",
                },
            ),
            "eta_rho": (
                ["eta_rho"],
                np.arange(Y, dtype="float32"),
                {
                    "long_name": "y-dimension of the grid",
                    "standard_name": "y_grid_index",
                    "axis": "Y",
                    "c_grid_dynamic_range": f"2:{Y}",
                },
            ),
            "eta_v": (
                ["eta_v"],
                np.arange(Y - 1, dtype="float32"),
                {
                    "long_name": "y-dimension of the grid at v location",
                    "standard_name": "x_grid_index_at_v_location",
                    "axis": "Y",
                    "c_grid_axis_shift": 0.5,
                    "c_grid_dynamic_range": f"2:{Y - 1}",
                },
            ),
            "xi_rho": (
                ["xi_rho"],
                np.arange(X, dtype="float32"),
                {
                    "long_name": "x-dimension of the grid",
                    "standard_name": "x_grid_index",
                    "axis": "X",
                    "c_grid_dynamic_range": f"2:{X}",
                },
            ),
            "xi_u": (
                ["xi_u"],
                np.arange(X - 1, dtype="float32"),
                {
                    "long_name": "x-dimension of the grid at u location",
                    "standard_name": "x_grid_index_at_u_location",
                    "axis": "X",
                    "c_grid_axis_shift": 0.5,
                    "c_grid_dynamic_range": f"2:{X - 1}",
                },
            ),
            "x_rho": (
                ["eta_rho", "xi_rho"],
                np.tile(np.linspace(-179, 179, X, endpoint=False), (Y, 1)),  # note that this is not curvilinear
                {
                    "long_name": "x-locations of RHO-points",
                    "units": "meter",
                    "standard_name": "plane_x_coordinate",
                    "field": "x_rho, scalar",
                },
            ),
            "y_rho": (
                ["eta_rho", "xi_rho"],
                np.tile(np.linspace(-89, 89, Y), (X, 1)).T,  # note that this is not curvilinear
                {
                    "long_name": "y-locations of RHO-points",
                    "units": "meter",
                    "standard_name": "plane_y_coordinate",
                    "field": "y_rho, scal",
                },
            ),
        },
    )


datasets = {
    "ds_copernicusmarine": _copernicusmarine(),
    "ds_copernicusmarine_waves": _copernicusmarine_waves(),
    "ds_NEMO_MOI_U": _NEMO_MOI_U(),
    "ds_NEMO_MOI_V": _NEMO_MOI_V(),
    "ds_CESM": _CESM(),
    "ds_MITgcm_netcdf": _MITgcm_netcdf(),
    "ds_MITgcm_mds": _MITgcm_mds(),
    "ds_ERA5_wind": _ERA5_wind(),
    "ds_FES_tides": _FES_tides(),
    "ds_hycom_espc": _hycom_espc(),
    "ds_ecco4": _ecco4(),
    "ds_CROCO_idealized": _CROCO_idealized(),
}
