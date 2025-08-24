import pytest
import pandas as pd
import polars as pl
import pyarrow as pa
import numpy as np
from collections import deque
from decimal import Decimal
from PIL import Image, ImageDraw, ImageFont
from coded_flows.utils import convert_type
from coded_flows.types import (
    Date,
    Datetime,
    Time,
    IPvAnyAddress,
    IPvAnyInterface,
    IPvAnyNetwork,
    AnyUrl,
    PostgresDsn,
    Path,
    NewPath,
    FilePath,
    DirectoryPath,
    Color,
    Coordinate,
)


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
def sample_arrow_table():
    data = {"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]}
    table = pa.table(data)
    return table


@pytest.fixture
def sample_datadict():
    return {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}


@pytest.fixture
def sample_datarecords():
    return [
        {"A": 1, "B": 4, "C": 7},
        {"A": 2, "B": 5, "C": 8},
        {"A": 3, "B": 6, "C": 9},
    ]


@pytest.fixture
def sample_date():
    return Date(2020, 1, 1)


@pytest.fixture
def sample_datetime():
    return Datetime(2020, 1, 1, 20, 10, 0, 0)


@pytest.fixture
def sample_url():
    return AnyUrl("https://www.example.com/path/to/resource?query=param#fragment")


@pytest.fixture
def sample_postgres_dsn():
    return PostgresDsn("postgres://username:password@hostname:4000/database_name")


@pytest.fixture
def sample_coordinate():
    return Coordinate(41.40338, 2.17403)


@pytest.fixture
def sample_image():
    image = Image.new("RGB", (300, 300), color="white")

    draw = ImageDraw.Draw(image)

    draw.rectangle([50, 50, 250, 250], outline="black", width=3)
    draw.ellipse([75, 75, 225, 225], outline="blue", width=3)
    draw.text((100, 130), "Hello, Pillow!", fill="black")

    return image


# ==================================

# ============== Tests =============


@pytest.mark.parametrize("json_value_type", json_value_sample_data.keys())
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


def test_dataseries_to_tuple(sample_series):
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


def test_pl_dataseries_to_dataframe(sample_polars_series):
    try:
        output = convert_type(sample_polars_series, "DataSeries", "DataFrame")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> Dataframe: {e}")
    assert isinstance(output, pl.DataFrame)


def test_pl_dataseries_to_list(sample_polars_series):
    try:
        output = convert_type(sample_polars_series, "DataSeries", "List")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> List: {e}")
    assert isinstance(output, list)


def test_pl_dataseries_to_tuple(sample_polars_series):
    try:
        output = convert_type(sample_polars_series, "DataSeries", "Tuple")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> Tuple: {e}")
    assert isinstance(output, tuple)


def test_pl_dataseries_to_set(sample_polars_series):
    try:
        output = convert_type(sample_polars_series, "DataSeries", "Set")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> Set: {e}")
    assert isinstance(output, set)


def test_pl_dataseries_to_json(sample_polars_series):
    try:
        output = convert_type(sample_polars_series, "DataSeries", "Json")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> Json: {e}")
    assert isinstance(output, str)


def test_pl_dataseries_to_numpy(sample_polars_series):
    try:
        output = convert_type(sample_polars_series, "DataSeries", "NDArray")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataSeries -> NDArray: {e}")
    assert isinstance(output, np.ndarray)


def test_pl_dataseries_to_arrow(sample_polars_series):
    try:
        output = convert_type(sample_polars_series, "DataSeries", "ArrowTable")
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


def test_dataframe_to_datadict(sample_dataframe):
    try:
        output = convert_type(sample_dataframe, "DataFrame", "DataDict")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> DataDict: {e}")
    assert all(isinstance(k, str) and isinstance(v, list) for k, v in output.items())


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


def test_pl_dataframe_to_datarecords(sample_polars_dataframe):
    try:
        output = convert_type(sample_polars_dataframe, "DataFrame", "DataRecords")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> DataRecords: {e}")
    assert isinstance(output, list) and all(isinstance(item, dict) for item in output)


def test_pl_dataframe_to_list(sample_polars_dataframe):
    try:
        output = convert_type(sample_polars_dataframe, "DataFrame", "List")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> List: {e}")
    assert isinstance(output, list)


def test_pl_dataframe_to_dict(sample_polars_dataframe):
    try:
        output = convert_type(sample_polars_dataframe, "DataFrame", "Dict")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> Dict: {e}")
    assert isinstance(output, dict)


def test_pl_dataframe_to_datadict(sample_polars_dataframe):
    try:
        output = convert_type(sample_polars_dataframe, "DataFrame", "DataDict")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> DataDict: {e}")
    assert all(isinstance(k, str) and isinstance(v, list) for k, v in output.items())


def test_pl_dataframe_to_json(sample_polars_dataframe):
    try:
        output = convert_type(sample_polars_dataframe, "DataFrame", "Json")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> Json: {e}")
    assert isinstance(output, str)


def test_pl_dataframe_to_numpy(sample_polars_dataframe):
    try:
        output = convert_type(sample_polars_dataframe, "DataFrame", "NDArray")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> NDArray: {e}")
    assert isinstance(output, np.ndarray)


def test_pl_dataframe_to_arrow(sample_polars_dataframe):
    try:
        output = convert_type(sample_polars_dataframe, "DataFrame", "ArrowTable")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> ArrowTable: {e}")
    assert isinstance(output, pa.Table)


def test_pl_lazy_dataframe_to_datarecords(sample_polars_lazyframe):
    try:
        output = convert_type(sample_polars_lazyframe, "DataFrame", "DataRecords")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> DataRecords: {e}")
    assert isinstance(output, list) and all(isinstance(item, dict) for item in output)


def test_pl_lazy_dataframe_to_list(sample_polars_lazyframe):
    try:
        output = convert_type(sample_polars_lazyframe, "DataFrame", "List")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> List: {e}")
    assert isinstance(output, list)


def test_pl_lazy_dataframe_to_dict(sample_polars_lazyframe):
    try:
        output = convert_type(sample_polars_lazyframe, "DataFrame", "Dict")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> Dict: {e}")
    assert isinstance(output, dict)


def test_pl_lazy_dataframe_to_datadict(sample_polars_lazyframe):
    try:
        output = convert_type(sample_polars_lazyframe, "DataFrame", "DataDict")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> DataDict: {e}")
    assert all(isinstance(k, str) and isinstance(v, list) for k, v in output.items())


def test_pl_lazy_dataframe_to_json(sample_polars_lazyframe):
    try:
        output = convert_type(sample_polars_lazyframe, "DataFrame", "Json")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> Json: {e}")
    assert isinstance(output, str)


def test_pl_lazy_dataframe_to_numpy(sample_polars_lazyframe):
    try:
        output = convert_type(sample_polars_lazyframe, "DataFrame", "NDArray")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> NDArray: {e}")
    assert isinstance(output, np.ndarray)


def test_pl_lazy_dataframe_to_arrow(sample_polars_lazyframe):
    try:
        output = convert_type(sample_polars_lazyframe, "DataFrame", "ArrowTable")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataFrame -> ArrowTable: {e}")
    assert isinstance(output, pa.Table)


def test_arrow_to_datarecords(sample_arrow_table):
    try:
        output = convert_type(sample_arrow_table, "ArrowTable", "DataRecords")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for ArrowTable -> DataRecords: {e}")
    assert isinstance(output, list) and all(isinstance(item, dict) for item in output)


def test_arrow_to_list(sample_arrow_table):
    try:
        output = convert_type(sample_arrow_table, "ArrowTable", "List")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for ArrowTable -> List: {e}")
    assert isinstance(output, list)


def test_arrow_to_dict(sample_arrow_table):
    try:
        output = convert_type(sample_arrow_table, "ArrowTable", "Dict")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for ArrowTable -> Dict: {e}")
    assert isinstance(output, dict)


def test_arrow_to_json(sample_arrow_table):
    try:
        output = convert_type(sample_arrow_table, "ArrowTable", "Json")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for ArrowTable -> Json: {e}")
    assert isinstance(output, str)


def test_arrow_to_numpy(sample_arrow_table):
    try:
        output = convert_type(sample_arrow_table, "ArrowTable", "NDArray")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for ArrowTable -> NDArray: {e}")
    assert isinstance(output, np.ndarray)


def test_arrow_to_dataframe(sample_arrow_table):
    try:
        output = convert_type(sample_arrow_table, "ArrowTable", "DataFrame")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for ArrowTable -> DataFrame: {e}")
    assert isinstance(output, pd.DataFrame)


def test_arrow_to_datadict(sample_arrow_table):
    try:
        output = convert_type(sample_arrow_table, "ArrowTable", "DataDict")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for ArrowTable -> DataDict: {e}")
    assert all(isinstance(k, str) and isinstance(v, list) for k, v in output.items())


def test_numpy_to_list(sample_numpy_array):
    try:
        output = convert_type(sample_numpy_array, "NDArray", "List")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for NDArray -> List: {e}")
    assert isinstance(output, list)


def test_numpy_to_json(sample_numpy_array):
    try:
        output = convert_type(sample_numpy_array, "NDArray", "Json")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for NDArray -> Json: {e}")
    assert isinstance(output, str)


def test_datadict_to_dataframe(sample_datadict):
    try:
        output = convert_type(sample_datadict, "DataDict", "DataFrame")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataDict -> DataFrame: {e}")
    assert isinstance(output, pd.DataFrame)


def test_datadict_to_arrow(sample_datadict):
    try:
        output = convert_type(sample_datadict, "DataDict", "ArrowTable")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataDict -> ArrowTable: {e}")
    assert isinstance(output, pa.Table)


def test_datarecords_to_dataframe(sample_datarecords):
    try:
        output = convert_type(sample_datarecords, "DataRecords", "DataFrame")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for DataRecords -> DataFrame: {e}")
    assert isinstance(output, pd.DataFrame)


def test_str_to_base64str():
    txt = "example"
    txt64 = "ZXhhbXBsZQ=="

    try:
        output = convert_type(txt, "Str", "Base64Str")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Str -> Base64Str: {e}")
    assert isinstance(output, str)
    assert output == txt64


def test_str_to_base64bytes():
    txt = "example"
    txt64 = b"ZXhhbXBsZQ=="

    try:
        output = convert_type(txt, "Str", "Base64Bytes")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Str -> Base64Bytes: {e}")
    assert isinstance(output, bytes)
    assert output == txt64


def test_str_to_bytes():
    txt = "example"

    try:
        output = convert_type(txt, "Str", "Bytes")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Str -> Bytes: {e}")
    assert isinstance(output, bytes)


def test_base64str_to_base64bytes():
    txt64 = "ZXhhbXBsZQ=="

    try:
        output = convert_type(txt64, "Base64Str", "Base64Bytes")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Base64Str -> Base64Bytes: {e}")
    assert isinstance(output, bytes)


def test_base64bytes_to_base64str():
    txt64 = b"ZXhhbXBsZQ=="

    try:
        output = convert_type(txt64, "Base64Bytes", "Base64Str")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Base64Bytes -> Base64Str: {e}")
    assert isinstance(output, str)


def test_bool_to_numeric():
    num_types = {"Int": int, "Float": float, "Complex": complex, "Decimal": Decimal}
    value = True

    try:
        for num_type_name, num_type in num_types.items():
            output = convert_type(value, "Bool", num_type_name)
            assert isinstance(output, num_type)
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Bool: {e}")


def test_datetime_to_time(sample_datetime):
    try:
        output = convert_type(sample_datetime, "Datetime", "Time")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Datetime -> Time: {e}")
    assert isinstance(output, Time)


def test_date_to_time(sample_date):
    try:
        output = convert_type(sample_date, "Date", "Time")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Date -> Time: {e}")
    assert isinstance(output, Time)


def test_date_to_datetime(sample_date):
    try:
        output = convert_type(sample_date, "Date", "Datetime")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Date -> Datetime: {e}")
    assert isinstance(output, Datetime)


def test_to_str():
    num_types = {
        "Int": 4,
        "Float": 3.14,
        "Decimal": Decimal(2.9),
        "IPvAnyAddress": IPvAnyAddress("192.168.1.1"),
        "IPvAnyInterface": IPvAnyInterface("192.168.1.1/24"),
        "IPvAnyNetwork": IPvAnyNetwork("192.168.1.1/32"),
        "Path": Path("/home/user/file.txt"),
        "NewPath": NewPath("/home/user/file.txt"),
        "FilePath": FilePath("/home/user/file.txt"),
        "DirectoryPath": DirectoryPath("/home/user/"),
    }

    try:
        for num_type_name, value in num_types.items():
            output = convert_type(value, num_type_name, "Str")
            assert isinstance(output, str)
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Numeric -> Str: {e}")


def test_list_to_types():
    output_types = {"Tuple": tuple, "Deque": deque, "Set": set, "FrozenSet": frozenset}
    value = [1, "56", 7, 2j]

    try:
        for output_type_name, output_type in output_types.items():
            output = convert_type(value, "List", output_type_name)
            assert isinstance(output, output_type)
    except TypeError as e:
        pytest.fail(f"TypeError encountered for List: {e}")


def test_tuple_to_types():
    output_types = {"List": list, "Deque": deque, "Set": set, "FrozenSet": frozenset}
    value = (1, "56", 7, 2j)

    try:
        for output_type_name, output_type in output_types.items():
            output = convert_type(value, "Tuple", output_type_name)
            assert isinstance(output, output_type)
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Tuple: {e}")


def test_deque_to_types():
    output_types = {"List": list, "Tuple": tuple, "Set": set, "FrozenSet": frozenset}
    value = deque([1, "56", 7, 2j])

    try:
        for output_type_name, output_type in output_types.items():
            output = convert_type(value, "Deque", output_type_name)
            assert isinstance(output, output_type)
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Deque: {e}")


def test_set_to_types():
    output_types = {
        "List": list,
        "Tuple": tuple,
        "Deque": deque,
        "FrozenSet": frozenset,
    }
    value = {1, "56", 7, 2j}

    try:
        for output_type_name, output_type in output_types.items():
            output = convert_type(value, "Set", output_type_name)
            assert isinstance(output, output_type)
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Set: {e}")


def test_frozenset_to_types():
    output_types = {
        "List": list,
        "Tuple": tuple,
        "Deque": deque,
        "Set": set,
    }
    value = frozenset([1, "56", 7, 2j])

    try:
        for output_type_name, output_type in output_types.items():
            output = convert_type(value, "FrozenSet", output_type_name)
            assert isinstance(output, output_type)
    except TypeError as e:
        pytest.fail(f"TypeError encountered for FrozenSet: {e}")


def test_url_to_str(sample_url):
    try:
        output = convert_type(sample_url, "AnyUrl", "Str")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for AnyUrl -> Str: {e}")
    assert isinstance(output, str)


def test_dsn_to_str(sample_postgres_dsn):
    try:
        output = convert_type(sample_postgres_dsn, "PostgresDsn", "Str")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for PostgresDsn -> Str: {e}")
    assert isinstance(output, str)


def test_color_to_str():
    value = Color((250, 0, 0))
    try:
        output = convert_type(value, "Color", "Str")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Color -> Str: {e}")
    assert isinstance(output, str)


def test_coordinate_to_list(sample_coordinate):
    try:
        output = convert_type(sample_coordinate, "Coordinate", "List")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Coordinate -> List: {e}")
    assert isinstance(output, list)
    assert len(output) == 2


def test_coordinate_to_tuple(sample_coordinate):
    try:
        output = convert_type(sample_coordinate, "Coordinate", "Tuple")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for Coordinate -> Tuple: {e}")
    assert isinstance(output, tuple)
    assert len(output) == 2


def test_image_to_numpy(sample_image):
    try:
        output = convert_type(sample_image, "PILImage", "NDArray")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for PILImage -> NDArray: {e}")
    assert isinstance(output, np.ndarray)


def test_image_to_bytes(sample_image):
    try:
        output = convert_type(sample_image, "PILImage", "Bytes")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for PILImage -> Bytes: {e}")
    assert isinstance(output, bytes)


def test_image_to_str(sample_image):
    try:
        output = convert_type(sample_image, "PILImage", "Str")
    except TypeError as e:
        pytest.fail(f"TypeError encountered for PILImage -> Str: {e}")
    assert isinstance(output, str)
