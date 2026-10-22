import pytest

from parcels import FieldSet
from parcels._datasets.structured.generic import datasets as datasets_structured


def pytest_addoption(parser: pytest.Parser):
    """Add command-line flags for pytest."""
    parser.addoption("--run-flaky-tests", action="store_true", help="runs flaky tests")
    parser.addoption(
        "--run-validation-tests",
        action="store_true",
        help="runs validation tests",
    )


def pytest_runtest_setup(item):
    if "flaky" in item.keywords and not item.config.getoption("--run-flaky-tests"):
        pytest.skip("set --run-flaky-tests to run flaky tests")
    if "validation" in item.keywords and not item.config.getoption("--run-validation-tests"):
        pytest.skip("set --run-validation-tests to run validation tests")


@pytest.fixture
def tmp_parquet(tmp_path):
    return tmp_path / "tmp.parquet"


@pytest.fixture
def tmp_zarr(tmp_path):
    return tmp_path / "tmp.zarr"


@pytest.fixture
def fieldset() -> FieldSet:
    """FieldSet with U and V"""
    ds = datasets_structured["ds_2d_left"].copy()
    ds = ds[["U_A_grid", "V_A_grid", "grid"]].rename(
        {
            "U_A_grid": "U",
            "V_A_grid": "V",
        }
    )
    return FieldSet.from_sgrid_conventions(ds)
