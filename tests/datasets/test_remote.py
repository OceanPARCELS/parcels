import pytest
import requests
import xarray as xr

import parcels._datasets.remote as remote


@pytest.fixture(scope="function", autouse=True)
def tmp_path_parcels_example_data(monkeypatch, tmp_path):
    monkeypatch.setenv("PARCELS_EXAMPLE_DATA", str(tmp_path))
    return tmp_path


@pytest.mark.flaky
@pytest.mark.parametrize("url", [remote._ODIE.get_url(filename) for filename in remote._ODIE.registry.keys()])
def test_pooch_registry_url_reponse(url):
    response = requests.head(url)
    assert not (400 <= response.status_code < 600)


def test_open_dataset_non_existing():
    with pytest.raises(ValueError, match="Dataset.*not found"):
        remote.open_remote_dataset("non_existing_dataset")


@pytest.mark.flaky
@pytest.mark.parametrize("name", remote.list_remote_datasets(scope="open"))
def test_open_dataset(name):
    ds = remote.open_remote_dataset(name)
    assert isinstance(ds, xr.Dataset)


@pytest.mark.flaky
@pytest.mark.parametrize("name", remote.list_remote_datasets(scope="download_only"))
def test_download_dataset(name):
    files = remote.get_remote_dataset(name)
    assert all(isinstance(f, str) for f in files)


@pytest.mark.flaky
@pytest.mark.parametrize("name", ["SWASH_data/data"])
def test_error_on_download_scope(name):
    with pytest.raises(ValueError, match="does not have Scope 'open'"):
        remote.open_remote_dataset(name)


@pytest.mark.parametrize("name", remote.list_remote_datasets())
def test_dataset_keys(name):
    assert not name.endswith((".zip", ".nc")), "Dataset name should not have suffix"


def test_list_datasets():
    tutorial_datasets = set(remote.list_remote_datasets("tutorial"))
    testing_datasets = set(remote.list_remote_datasets("testing"))
    all_datasets = set(remote.list_remote_datasets("any"))
    assert tutorial_datasets.issubset(all_datasets)
    assert testing_datasets.issubset(all_datasets)
    assert tutorial_datasets | testing_datasets == all_datasets
