import numpy as np
import pandas as pd
import polars as pl
import pytest
from PIL import Image
import pyarrow as pa
from pydantic_core import ValidationError
from coded_flows.types import (
    DataFrame,
    DataSeries,
    NDArray,
    ArrowTable,
    PILImage,
    is_valid_value_type,
)


@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample pandas DataFrame."""
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})


@pytest.fixture
def sample_series():
    """Fixture to create a sample pandas Series."""
    return pd.Series([1, 2, 3, 4, 5])


@pytest.fixture
def sample_polars_dataframe():
    """Fixture to create a sample Polars DataFrame."""
    return pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})


@pytest.fixture
def sample_polars_series():
    """Fixture to create a sample Polars Series."""
    return pl.Series([1, 2, 3, 4, 5])


@pytest.fixture
def sample_polars_lazyframe():
    """Fixture to create a sample Polars LazyFrame."""
    return pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}).lazy()


@pytest.fixture
def sample_numpy_array():
    """Fixture to create a sample numpy array."""
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def sample_pillow_image():
    # Create a new image with RGB mode and white background
    size = (100, 100)  # Image size
    color = (255, 255, 255)  # White
    image = Image.new("RGB", size, color)
    return image


@pytest.fixture
def sample_pyarrow_table():
    # Create a simple PyArrow table from a dictionary of lists
    data = {"column1": [1, 2, 3], "column2": ["a", "b", "c"]}
    table = pa.Table.from_pydict(data)
    return table


# DataFrame
def test_valid_dataframe_type(sample_dataframe):
    try:
        is_valid_value_type(sample_dataframe, DataFrame)
    except ValidationError:
        pytest.fail("ValidationError was raised for a valid type match of a DataFrame")


def test_valid_pl_dataframe_type(sample_polars_dataframe):
    try:
        is_valid_value_type(sample_polars_dataframe, DataFrame)
    except ValidationError:
        pytest.fail("ValidationError was raised for a valid type match of a DataFrame")


def test_valid_pl_lazy_dataframe_type(sample_polars_lazyframe):
    try:
        is_valid_value_type(sample_polars_lazyframe, DataFrame)
    except ValidationError:
        pytest.fail("ValidationError was raised for a valid type match of a DataFrame")


def test_invalid_dataframe_type():
    with pytest.raises(ValidationError):
        is_valid_value_type(3, DataFrame)


# DataSeries
def test_valid_series_type(sample_series):
    try:
        is_valid_value_type(sample_series, DataSeries)
    except ValidationError:
        pytest.fail("ValidationError was raised for a valid type match of a DataSeries")


def test_valid_pl_series_type(sample_polars_series):
    try:
        is_valid_value_type(sample_polars_series, DataSeries)
    except ValidationError:
        pytest.fail("ValidationError was raised for a valid type match of a DataSeries")


def test_invalid_series_type():
    with pytest.raises(ValidationError):
        is_valid_value_type(3, DataSeries)


# NDArray
def test_valid_ndarray_type(sample_numpy_array):
    try:
        is_valid_value_type(sample_numpy_array, NDArray)
    except ValidationError:
        pytest.fail("ValidationError was raised for a valid type match of a NDArray")


def test_invalid_ndarray_type():
    with pytest.raises(ValidationError):
        is_valid_value_type(3, NDArray)


# Arrow
def test_valid_arrow_type(sample_pyarrow_table):
    try:
        is_valid_value_type(sample_pyarrow_table, ArrowTable)
    except ValidationError:
        pytest.fail("ValidationError was raised for a valid type match of a ArrowTable")


def test_invalid_arrow_type():
    with pytest.raises(ValidationError):
        is_valid_value_type(3, ArrowTable)


# Pillow
def test_valid_pillow_type(sample_pillow_image):
    try:
        is_valid_value_type(sample_pillow_image, PILImage)
    except ValidationError:
        pytest.fail("ValidationError was raised for a valid type match of a PILImage")


def test_invalid_pillow_type():
    with pytest.raises(ValidationError):
        is_valid_value_type(3, PILImage)


if __name__ == "__main__":
    pytest.main()
