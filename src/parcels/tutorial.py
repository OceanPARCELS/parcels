from parcels._datasets.remote import _DATA_HOME
from parcels._datasets.remote import list_remote_datasets as _list_remote_datasets
from parcels._datasets.remote import open_remote_dataset as _open_remote_dataset

__all__ = ["cache_path", "list_datasets", "open_dataset"]


def list_datasets() -> list[str]:
    """List the available tutorial datasets.

    Use :func:`open_dataset` to download and open one of the datasets.

    Returns
    -------
    datasets : list of str
        The names of the available datasets matching the given purpose.
    """
    return _list_remote_datasets(purpose="tutorial")


def open_dataset(name: str):
    """Download and open a tutorial dataset as an :class:`xarray.Dataset`.

    Use :func:`list_datasets` to see the available dataset names.

    Parameters
    ----------
    name : str
        Name of the dataset to open. Must be one of the keys returned by
        :func:`list_datasets`.

    Returns
    -------
    xarray.Dataset
        The requested dataset.
    """
    return _open_remote_dataset(name, purpose="tutorial")


cache_path = _DATA_HOME.resolve()
