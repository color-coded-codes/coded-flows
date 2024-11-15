import pytest
import json
import pandas as pd
import pyarrow as pa
import numpy as np
from ..coded_flows.utils import convert_type


json_value_sample_data = {
    "List": [1, 2, 3],
    "Dict": {"key": "value"},
    "Str": "example string",
    "Base64Str": "ZXhhbXBsZQ==",
    "CountryAlpha2": "US",
    "CountryAlpha3": "USA",
    "CountryNumericCode": "840",
    "CountryShortName": "United States",
    "EmailStr": "test@example.com",
    "Currency": "USD",
    "MacAddress": "00:1B:44:11:3A:B7",
    "Bool": True,
    "Int": 42,
    "Float": 3.14159,
    "Null": None,
}


# ============ Fixtures ============
@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample pandas DataFrame."""
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})


@pytest.fixture
def sample_series():
    """Fixture to create a sample pandas Series."""
    return pd.Series([1, 2, 3, 4, 5])


@pytest.fixture
def sample_numpy_array():
    """Fixture to create a sample numpy array."""
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def arrow_table():
    data = {"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]}
    table = pa.table(data)
    return table


# ==================================

# ============== Tests =============


@pytest.mark.parametrize("json_type", json_value_sample_data.keys())
def test_json_values_to_json(json_value_type):
    value = json_value_sample_data[json_value_type]

    try:
        converted_value = convert_type(value, json_value_type, "Json")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for {json_value_type}: {e}")

    assert isinstance(converted_value, str)


def test_dataseries_to_dataframe(sample_series):
    try:
        output = convert_type(sample_series, "DataSeries", "DataFrame")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> Dataframe: {e}")
    assert isinstance(output, pd.DataFrame)


def test_dataseries_to_list(sample_series):
    try:
        output = convert_type(sample_series, "DataSeries", "List")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> List: {e}")
    assert isinstance(output, list)


def test_dataseries_to_list(sample_series):
    try:
        output = convert_type(sample_series, "DataSeries", "Tuple")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> Tuple: {e}")
    assert isinstance(output, tuple)


def test_dataseries_to_set(sample_series):
    try:
        output = convert_type(sample_series, "DataSeries", "Set")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> Set: {e}")
    assert isinstance(output, set)


def test_dataseries_to_json(sample_series):
    try:
        output = convert_type(sample_series, "DataSeries", "Json")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> Json: {e}")
    assert isinstance(output, str)


def test_dataseries_to_numpy(sample_series):
    try:
        output = convert_type(sample_series, "DataSeries", "NDArray")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> NDArray: {e}")
    assert isinstance(output, np.ndarray)


def test_dataseries_to_arrow(sample_series):
    try:
        output = convert_type(sample_series, "DataSeries", "ArrowTable")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> ArrowTable: {e}")
    assert isinstance(output, pa.Table)


def test_dataframe_to_datarecords(sample_dataframe):
    try:
        output = convert_type(sample_dataframe, "DataFrame", "DataRecords")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> DataRecords: {e}")
    assert isinstance(output, list) and all(isinstance(item, dict) for item in output)


def test_dataframe_to_list(sample_dataframe):
    try:
        output = convert_type(sample_dataframe, "DataFrame", "List")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> List: {e}")
    assert isinstance(output, list)


def test_dataframe_to_dict(sample_dataframe):
    try:
        output = convert_type(sample_dataframe, "DataFrame", "Dict")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> Dict: {e}")
    assert isinstance(output, dict)


def test_dataframe_to_json(sample_dataframe):
    try:
        output = convert_type(sample_dataframe, "DataFrame", "Json")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> Json: {e}")
    assert isinstance(output, str)


def test_dataframe_to_numpy(sample_dataframe):
    try:
        output = convert_type(sample_dataframe, "DataFrame", "NDArray")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> NDArray: {e}")
    assert isinstance(output, np.ndarray)


def test_dataframe_to_arrow(sample_dataframe):
    try:
        output = convert_type(sample_dataframe, "DataFrame", "ArrowTable")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> ArrowTable: {e}")
    assert isinstance(output, pa.Table)


# DataRecords
def test_arrow_to_arrow(sample_dataframe):
    try:
        output = convert_type(sample_dataframe, "ArrowTable", "NDArray")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for ArrowTable -> NDArray: {e}")
    assert isinstance(output, np.ndarray)


# List
# Dict
# Json
# NDArray
# DataFrame
