import abc
import enum
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pooch
import xarray as xr
from zarr.storage import ZipStore

# When modifying existing datasets in a backwards incompatible way,
# make a new release in the repo and update the DATA_REPO_TAG to the new tag
_DATA_REPO_TAG = "main"

_DATA_URL = f"https://github.com/Parcels-code/parcels-data/raw/{_DATA_REPO_TAG}"


def _get_data_home() -> Path:
    data_home = os.environ.get("PARCELS_EXAMPLE_DATA")
    if data_home is None:
        data_home = pooch.os_cache("parcels")
    return Path(data_home)


_DATA_HOME = _get_data_home()

# See instructions at https://github.com/Parcels-code/parcels-data for adding new datasets
_ODIE_REGISTRY_FILES: list[str] = (
    # These datasets are from v3 and before of Parcels, where we just used netcdf files
    # [
    #     "data/MovingEddies_data/moving_eddiesP.nc",
    #     "data/MovingEddies_data/moving_eddiesU.nc",
    #     "data/MovingEddies_data/moving_eddiesV.nc",
    # ]
    ["data/MITgcm_example_data/mitgcm_UV_surface_zonally_reentrant.nc"]
    + ["data/OFAM_example_data/OFAM_simple_U.nc", "data/OFAM_example_data/OFAM_simple_V.nc"]
    # + [
    #     "data/Peninsula_data/peninsulaU.nc",
    #     "data/Peninsula_data/peninsulaV.nc",
    #     "data/Peninsula_data/peninsulaP.nc",
    #     "data/Peninsula_data/peninsulaT.nc",
    # ]
    # + [
    #     f"data/GlobCurrent_example_data/{date.strftime('%Y%m%d')}000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc"
    #     for date in ([datetime(2002, 1, 1) + timedelta(days=x) for x in range(0, 365)] + [datetime(2003, 1, 1)])
    # ]
    + [
        "data/CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_uo-vo_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc",
        "data/CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m_so_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc",
        "data/CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m_thetao_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc",
        "data/CopernicusMarine_data_for_Argo_tutorial/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_multi-vars_31.06E-32.94E_32.94S-30.06S_2024-01-01-2024-02-01.nc",
    ]
    + ["data/CopernicusMarine_data_for_stuck_particles_tutorial/cmems_mod_glo_phy_my_0.083deg_P1D-m_NL.nc"]
    # + [
    #     "data/DecayingMovingEddy_data/decaying_moving_eddyU.nc",
    #     "data/DecayingMovingEddy_data/decaying_moving_eddyV.nc",
    # ]
    + [
        "data/Delft3D_data/Rotterdam_tiny.nc",
    ]
    + [
        "data/FESOM_periodic_channel/fesom_channel.nc",
        "data/FESOM_periodic_channel/u.fesom_channel.nc",
        "data/FESOM_periodic_channel/v.fesom_channel.nc",
        "data/FESOM_periodic_channel/w.fesom_channel.nc",
    ]
    + [
        "data/SCHISM_LakeOntario/out2d.schism_lake_ontario.nc",
        "data/SCHISM_LakeOntario/horizontalVelX.schism_lake_ontario.nc",
        "data/SCHISM_LakeOntario/horizontalVelY.schism_lake_ontario.nc",
    ]
    + [
        "data/NemoCurvilinear_data/U_purely_zonal-ORCA025_grid_U.nc4",
        "data/NemoCurvilinear_data/V_purely_zonal-ORCA025_grid_V.nc4",
        "data/NemoCurvilinear_data/mesh_mask.nc4",
    ]
    + [
        "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000104d05U.nc",
        "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000109d05U.nc",
        "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000114d05U.nc",
        # "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000119d05U.nc",
        # "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000124d05U.nc",
        # "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000129d05U.nc",
        "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000104d05V.nc",
        "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000109d05V.nc",
        "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000114d05V.nc",
        # "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000119d05V.nc",
        # "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000124d05V.nc",
        # "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000129d05V.nc",
        "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000104d05W.nc",
        "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000109d05W.nc",
        "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000114d05W.nc",
        # "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000119d05W.nc",
        # "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000124d05W.nc",
        # "data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_20000129d05W.nc",
        "data/NemoNorthSeaORCA025-N006_data/coordinates.nc",
    ]
    + [
        "data/POPSouthernOcean_data/t.x1_SAMOC_flux.169000.nc",
        "data/POPSouthernOcean_data/t.x1_SAMOC_flux.169001.nc",
        "data/POPSouthernOcean_data/t.x1_SAMOC_flux.169002.nc",
        "data/POPSouthernOcean_data/t.x1_SAMOC_flux.169003.nc",
        "data/POPSouthernOcean_data/t.x1_SAMOC_flux.169004.nc",
        "data/POPSouthernOcean_data/t.x1_SAMOC_flux.169005.nc",
    ]
    + [
        "data/SWASH_data/field_0065532.nc",
        "data/SWASH_data/field_0065537.nc",
        "data/SWASH_data/field_0065542.nc",
        "data/SWASH_data/field_0065548.nc",
        "data/SWASH_data/field_0065552.nc",
        "data/SWASH_data/field_0065557.nc",
    ]
    # + [f"data/WOA_data/woa18_decav_t{m:02d}_04.nc" for m in range(1, 13)]
    + ["data/CROCOidealized_data/CROCO_idealized.nc"]
    # These datasets are from v4 of Parcels where we're opting for Zipped zarr datasets
    + [
        "data-zarr/Benchmarks_FESOM2-baroclinic-gyre/data.zip",
        "data-zarr/Benchmarks_FESOM2-baroclinic-gyre/grid.zip",
    ]
    + [
        "data-zarr/Benchmarks_MOi_data_metadata-only/U.zip",
        "data-zarr/Benchmarks_MOi_data_metadata-only/V.zip",
        "data-zarr/Benchmarks_MOi_data_metadata-only/W.zip",
        "data-zarr/Benchmarks_MOi_data_metadata-only/mesh.zip",
    ]
)

_ODIE = pooch.create(
    path=_DATA_HOME,
    base_url=_DATA_URL,
    registry={k: None for k in _ODIE_REGISTRY_FILES},
)


class _ParcelsDataset(abc.ABC):
    @abc.abstractmethod
    def open_dataset(self) -> xr.Dataset: ...


class _V3Dataset(_ParcelsDataset):
    def __init__(self, pup: pooch.Pooch, path_relative_to_pup: str, pre_decode_cf_callable=None):
        self.path_relative_to_root = path_relative_to_pup  # glob is allowed

        # Function to apply to the dataset before the decoding the CF variables
        self.pup = pup
        self.pre_decode_cf_callable: Callable[[xr.Dataset], xr.Dataset] | None = pre_decode_cf_callable

        first, second, *_ = path_relative_to_pup.split("/")
        self.v3_dataset_name = f"{first}/{second}"  # e.g., data/my_dataset

    def open_dataset(self) -> xr.Dataset:
        self.download_relevant_files()
        with xr.set_options(use_new_combine_kwarg_defaults=True):
            ds = xr.open_mfdataset(
                f"{self.pup.path}/{self.path_relative_to_root}",
                decode_cf=False,
                # options to open mfdataset https://github.com/Parcels-code/Parcels/pull/2574#discussion_r3073256988
                data_vars="minimal",
                coords="minimal",
                compat="override",
            )

        if self.pre_decode_cf_callable is not None:
            ds = self.pre_decode_cf_callable(ds)

        ds = xr.decode_cf(ds)
        return ds

    def download_relevant_files(self) -> None:
        for file in self.pup.registry:
            if self.v3_dataset_name in file:
                self.pup.fetch(file)
        return


class _ZarrZipDataset(_ParcelsDataset):
    def __init__(self, pup, path_relative_to_pup, zarr_format: Literal[2, 3] = 3):
        self.pup = pup
        self.path_relative_to_root = path_relative_to_pup
        self.zarr_format = zarr_format

    def open_dataset(self) -> xr.Dataset:
        self.pup.fetch(self.path_relative_to_root)
        return xr.open_zarr(ZipStore(Path(self.pup.path) / self.path_relative_to_root), zarr_format=self.zarr_format)


def _preprocess_drop_time_from_mesh1(ds: xr.Dataset) -> xr.Dataset:
    # For some reason on the mesh "NemoNorthSeaORCA025-N006_data/coordinates.nc" there are two time dimensions (of length 1). These dimension also has broken cf-time metadata
    # this fixes that
    return ds.isel(time=0).drop_vars(["time", "time_steps"])


def _preprocess_drop_time_from_mesh2(ds: xr.Dataset) -> xr.Dataset:
    # For some reason on the mesh "NemoCurvilinear_data_zonal/mesh_mask" there is a time dimension.
    return ds.isel(time=0).drop_vars(["time"])


def _preprocess_set_cf_calendar_360_day(ds: xr.Dataset) -> xr.Dataset:
    # For some reason "WOA_data/woa18_decav_t*_04.nc" looks to be simulation data using CF time (i.e., months of 30 days), however the calendar attribute isn't set.
    ds.time.attrs.update({"calendar": "360_day"})
    return ds


class _Purpose(enum.Enum):
    TESTING = "testing"
    TUTORIAL = "tutorial"


_TPurpose = Literal["testing", "tutorial"]

# The first here is a human readable key used to open datasets, with an object to open the datasets
# fmt: off
_DATASET_KEYS_AND_CONFIGS: dict[str, tuple[_ParcelsDataset, _Purpose]] = dict([
    # ("MovingEddies_data/P", (_V3Dataset(_ODIE,"data/MovingEddies_data/moving_eddiesP.nc"), _Purpose.TUTORIAL)),
    # ("MovingEddies_data/U", (_V3Dataset(_ODIE,"data/MovingEddies_data/moving_eddiesU.nc"), _Purpose.TUTORIAL)),
    # ("MovingEddies_data/V", (_V3Dataset(_ODIE,"data/MovingEddies_data/moving_eddiesV.nc"), _Purpose.TUTORIAL)),
    ("MITgcm_example_data/mitgcm_UV_surface_zonally_reentrant", (_V3Dataset(_ODIE,"data/MITgcm_example_data/mitgcm_UV_surface_zonally_reentrant.nc"), _Purpose.TUTORIAL)),
    # ("OFAM_example_data/U", (_V3Dataset(_ODIE,"data/OFAM_example_data/OFAM_simple_U.nc"), _Purpose.TUTORIAL)),
    # ("OFAM_example_data/V", (_V3Dataset(_ODIE,"data/OFAM_example_data/OFAM_simple_V.nc"), _Purpose.TUTORIAL)),
    # ("Peninsula_data/U", (_V3Dataset(_ODIE,"data/Peninsula_data/peninsulaU.nc"), _Purpose.TUTORIAL)),
    # ("Peninsula_data/V", (_V3Dataset(_ODIE,"data/Peninsula_data/peninsulaV.nc"), _Purpose.TUTORIAL)),
    # ("Peninsula_data/P", (_V3Dataset(_ODIE,"data/Peninsula_data/peninsulaP.nc"), _Purpose.TUTORIAL)),
    # ("Peninsula_data/T", (_V3Dataset(_ODIE,"data/Peninsula_data/peninsulaT.nc"), _Purpose.TUTORIAL)),
    # ("GlobCurrent_example_data/data", (_V3Dataset(_ODIE,"data/GlobCurrent_example_data/*000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc", pre_decode_cf_callable=patch_dataset_v4_compat), _Purpose.TUTORIAL)),
    ("CopernicusMarine_data_for_Argo_tutorial/data", (_V3Dataset(_ODIE,"data/CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-*.nc"), _Purpose.TUTORIAL)),
    ("CopernicusMarine_data_for_Argo_tutorial/data_wind", (_V3Dataset(_ODIE,"data/CopernicusMarine_data_for_Argo_tutorial/cmems_obs-wind_glo_phy*.nc"), _Purpose.TUTORIAL)),
    ("Delft3D_data/Rotterdam_tiny", (_V3Dataset(_ODIE,"data/Delft3D_data/Rotterdam_tiny.nc"), _Purpose.TUTORIAL)),
    ("CopernicusMarine_data_for_stuck_particles_tutorial/data", (_V3Dataset(_ODIE,"data/CopernicusMarine_data_for_stuck_particles_tutorial/cmems_mod_glo_phy_my_0.083deg_P1D-m_NL.nc"), _Purpose.TUTORIAL)),
    # ("DecayingMovingEddy_data/U", (_V3Dataset(_ODIE,"data/DecayingMovingEddy_data/decaying_moving_eddyU.nc"), _Purpose.TUTORIAL)),
    # ("DecayingMovingEddy_data/V", (_V3Dataset(_ODIE,"data/DecayingMovingEddy_data/decaying_moving_eddyV.nc"), _Purpose.TUTORIAL)),
    ("FESOM_periodic_channel/fesom_channel", (_V3Dataset(_ODIE,"data/FESOM_periodic_channel/fesom_channel.nc"), _Purpose.TUTORIAL)),
    ("FESOM_periodic_channel/u.fesom_channel", (_V3Dataset(_ODIE,"data/FESOM_periodic_channel/u.fesom_channel.nc"), _Purpose.TUTORIAL)),
    ("FESOM_periodic_channel/v.fesom_channel", (_V3Dataset(_ODIE,"data/FESOM_periodic_channel/v.fesom_channel.nc"), _Purpose.TUTORIAL)),
    ("FESOM_periodic_channel/w.fesom_channel", (_V3Dataset(_ODIE,"data/FESOM_periodic_channel/w.fesom_channel.nc"), _Purpose.TUTORIAL)),
    ("SCHISM_LakeOntario/out2d", (_V3Dataset(_ODIE,"data/SCHISM_LakeOntario/out2d.schism_lake_ontario.nc"), _Purpose.TUTORIAL)),
    ("SCHISM_LakeOntario/horizontalVelX", (_V3Dataset(_ODIE,"data/SCHISM_LakeOntario/horizontalVelX.schism_lake_ontario.nc"), _Purpose.TUTORIAL)),
    ("SCHISM_LakeOntario/horizontalVelY", (_V3Dataset(_ODIE,"data/SCHISM_LakeOntario/horizontalVelY.schism_lake_ontario.nc"), _Purpose.TUTORIAL)),
    ("NemoCurvilinear_data_zonal/U", (_V3Dataset(_ODIE,"data/NemoCurvilinear_data/U_purely_zonal-ORCA025_grid_U.nc4"), _Purpose.TUTORIAL)),
    ("NemoCurvilinear_data_zonal/V", (_V3Dataset(_ODIE,"data/NemoCurvilinear_data/V_purely_zonal-ORCA025_grid_V.nc4"), _Purpose.TUTORIAL)),
    ("NemoCurvilinear_data_zonal/mesh_mask", (_V3Dataset(_ODIE,"data/NemoCurvilinear_data/mesh_mask.nc4", _preprocess_drop_time_from_mesh2), _Purpose.TUTORIAL)),
    ("NemoNorthSeaORCA025-N006_data/U", (_V3Dataset(_ODIE,"data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_200001*05U.nc"), _Purpose.TUTORIAL)),
    ("NemoNorthSeaORCA025-N006_data/V", (_V3Dataset(_ODIE,"data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_200001*05V.nc"), _Purpose.TUTORIAL)),
    ("NemoNorthSeaORCA025-N006_data/W", (_V3Dataset(_ODIE,"data/NemoNorthSeaORCA025-N006_data/ORCA025-N06_200001*05W.nc"), _Purpose.TUTORIAL)),
    ("NemoNorthSeaORCA025-N006_data/mesh_mask", (_V3Dataset(_ODIE,"data/NemoNorthSeaORCA025-N006_data/coordinates.nc", _preprocess_drop_time_from_mesh1), _Purpose.TUTORIAL)),
    # "POPSouthernOcean_data/t.x1_SAMOC_flux.16900*.nc", # TODO v4: In v3 but should not be in v4 https://github.com/Parcels-code/Parcels/issues/2571#issuecomment-4214476973
    # ("SWASH_data/data", (_V3Dataset(_ODIE,"data/SWASH_data/field_00655*.nc"), _Purpose.TUTORIAL)),
    # ("WOA_data/data", (_V3Dataset(_ODIE,"data/WOA_data/woa18_decav_t*_04.nc", _preprocess_set_cf_calendar_360_day), _Purpose.TUTORIAL)),
    ("CROCOidealized_data/data", (_V3Dataset(_ODIE,"data/CROCOidealized_data/CROCO_idealized.nc"), _Purpose.TUTORIAL)),
] + [
    ("Benchmarks_FESOM2-baroclinic-gyre/data", (_ZarrZipDataset(_ODIE, 'data-zarr/Benchmarks_FESOM2-baroclinic-gyre/data.zip', zarr_format=2), _Purpose.TESTING)),
    ("Benchmarks_FESOM2-baroclinic-gyre/grid", (_ZarrZipDataset(_ODIE, 'data-zarr/Benchmarks_FESOM2-baroclinic-gyre/grid.zip', zarr_format=2),_Purpose.TESTING)),
    ("Benchmarks_MOi_data_metadata-only/U", (_ZarrZipDataset(_ODIE, "data-zarr/Benchmarks_MOi_data_metadata-only/U.zip"), _Purpose.TESTING)),
    ("Benchmarks_MOi_data_metadata-only/V", (_ZarrZipDataset(_ODIE, "data-zarr/Benchmarks_MOi_data_metadata-only/V.zip"), _Purpose.TESTING)),
    ("Benchmarks_MOi_data_metadata-only/W", (_ZarrZipDataset(_ODIE, "data-zarr/Benchmarks_MOi_data_metadata-only/W.zip"), _Purpose.TESTING)),
    ("Benchmarks_MOi_data_metadata-only/mesh", (_ZarrZipDataset(_ODIE, "data-zarr/Benchmarks_MOi_data_metadata-only/mesh.zip"), _Purpose.TESTING)),
])
# fmt: on


def list_remote_datasets(purpose: _TPurpose | Literal["any"] = "any") -> list[str]:
    """List the available remote datasets.

    Use :func:`open_dataset` to download and open one of the datasets.

    Parameters
    ----------
    purpose : {'any', 'testing', 'tutorial'}, optional
        Filter datasets by purpose. Use ``'any'`` (default) to return all
        datasets, ``'tutorial'`` for tutorial datasets, or ``'testing'`` for
        datasets used in tests.

    Returns
    -------
    datasets : list of str
        The names of the available datasets matching the given purpose.
    """
    if purpose == "any":
        return list(_DATASET_KEYS_AND_CONFIGS.keys())

    purpose_enum = _Purpose(purpose)
    return [k for (k, (_, p)) in _DATASET_KEYS_AND_CONFIGS.items() if p == purpose_enum]


def open_remote_dataset(name: str, purpose: _TPurpose | Literal["any"] = "any"):
    """Download and open a remote dataset as an :class:`xarray.Dataset`.

    Use :func:`list_datasets` to see the available dataset names.

    Parameters
    ----------
    name : str
        Name of the dataset to open. Must be one of the keys returned by
        :func:`list_datasets`.
    purpose : {'any', 'testing', 'tutorial'}, optional
        Purpose filter used to populate the error message when ``name`` is not
        found. Defaults to ``'any'``.

    Returns
    -------
    xarray.Dataset
        The requested dataset.
    """
    if name not in list_remote_datasets(purpose=purpose):
        raise ValueError(
            f"Dataset {name!r} not found. Available datasets are: " + ", ".join(list_remote_datasets(purpose=purpose))
        )

    dataset_config = _DATASET_KEYS_AND_CONFIGS[name][0]
    return dataset_config.open_dataset()
