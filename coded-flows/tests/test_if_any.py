import pytest
import pandas as pd
import polars as pl
import pyarrow as pa
import numpy as np
from PIL import Image
from collections import deque
from pathlib import Path
import uuid
from decimal import Decimal
from datetime import datetime, date, time, timedelta
from io import BytesIO, StringIO
import tempfile
import os

from coded_flows.utils import if_any


# Basic types tests
def test_none_returns_false():
    assert if_any(None) == False


def test_bool_values():
    assert if_any(True) == True
    assert if_any(False) == False


def test_numeric_values():
    """All numeric values should return True."""
    assert if_any(1) == True
    assert if_any(0) == True  # Note: differs from standard bool()
    assert if_any(-1) == True
    assert if_any(1.5) == True
    assert if_any(0.0) == True
    assert if_any(-2.5) == True
    assert if_any(complex(1, 2)) == True
    assert if_any(complex(0, 0)) == True


def test_decimal_values():
    assert if_any(Decimal("1.5")) == True
    assert if_any(Decimal("0")) == True
    assert if_any(Decimal("-1.5")) == True


def test_string_values():
    assert if_any("hello") == True
    assert if_any("") == False
    assert if_any(" ") == True  # Non-empty string
    assert if_any("0") == True


def test_bytes_values():
    assert if_any(b"hello") == True
    assert if_any(b"") == False
    assert if_any(bytearray(b"hello")) == True
    assert if_any(bytearray()) == False


# Collection types tests
def test_list_values():
    assert if_any([1, 2, 3]) == True
    assert if_any([]) == False
    assert if_any([None]) == True  # Non-empty list


def test_tuple_values():
    assert if_any((1, 2, 3)) == True
    assert if_any(()) == False
    assert if_any((None,)) == True


def test_set_values():
    assert if_any({1, 2, 3}) == True
    assert if_any(set()) == False
    assert if_any(frozenset([1, 2])) == True
    assert if_any(frozenset()) == False


def test_dict_values():
    assert if_any({"a": 1}) == True
    assert if_any({}) == False


def test_deque_values():
    assert if_any(deque([1, 2, 3])) == True
    assert if_any(deque()) == False


# Path objects tests
def test_existing_path():
    """Test with existing file path."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        assert if_any(tmp_path) == True
    finally:
        os.unlink(tmp_path)


def test_non_existing_path():
    non_existing_path = Path("/this/path/should/not/exist/12345")
    assert if_any(non_existing_path) == False


# Pandas tests
def test_pandas_series_with_values():
    series_with_values = pd.Series([1, 2, 3])
    assert if_any(series_with_values) == True


def test_pandas_empty_series():
    empty_series = pd.Series([], dtype=float)
    assert if_any(empty_series) == False


def test_pandas_series_all_nan():
    nan_series = pd.Series([np.nan, np.nan])
    assert if_any(nan_series) == False


def test_pandas_series_mixed_nan():
    mixed_series = pd.Series([1, np.nan, 3])
    assert if_any(mixed_series) == True


def test_pandas_dataframe_with_values():
    df_with_values = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    assert if_any(df_with_values) == True


def test_pandas_empty_dataframe():
    empty_df = pd.DataFrame()
    assert if_any(empty_df) == False


def test_pandas_dataframe_all_nan():
    nan_df = pd.DataFrame({"A": [np.nan, np.nan], "B": [np.nan, np.nan]})
    assert if_any(nan_df) == False


def test_pandas_dataframe_mixed_nan():
    mixed_df = pd.DataFrame({"A": [1, np.nan], "B": [np.nan, 4]})
    assert if_any(mixed_df) == True


# Polars tests
def test_polars_series_with_values():
    series_with_values = pl.Series([1, 2, 3])
    assert if_any(series_with_values) == True


def test_polars_empty_series():
    empty_series = pl.Series([], dtype=pl.Int64)
    assert if_any(empty_series) == False


def test_polars_series_all_null():
    null_series = pl.Series([None, None])
    assert if_any(null_series) == False


def test_polars_series_mixed_null():
    mixed_series = pl.Series([1, None, 3])
    assert if_any(mixed_series) == True


def test_polars_dataframe_with_values():
    df_with_values = pl.DataFrame({"A": [1, 2], "B": [3, 4]})
    assert if_any(df_with_values) == True


def test_polars_empty_dataframe():
    empty_df = pl.DataFrame()
    assert if_any(empty_df) == False


def test_polars_dataframe_all_null():
    null_df = pl.DataFrame({"A": [None, None], "B": [None, None]})
    assert if_any(null_df) == False


def test_polars_dataframe_mixed_null():
    mixed_df = pl.DataFrame({"A": [1, None], "B": [None, 4]})
    assert if_any(mixed_df) == True


# NumPy tests
def test_numpy_array_with_values():
    array_with_values = np.array([1, 2, 3])
    assert if_any(array_with_values) == True


def test_numpy_empty_array():
    empty_array = np.array([])
    assert if_any(empty_array) == False


def test_numpy_multidimensional_array():
    array_2d = np.array([[1, 2], [3, 4]])
    assert if_any(array_2d) == True


def test_numpy_empty_2d_array():
    empty_2d = np.array([]).reshape(0, 2)
    assert if_any(empty_2d) == False


# PyArrow tests
def test_pyarrow_table_with_data():
    table_with_data = pa.table({"A": [1, 2, 3], "B": [4, 5, 6]})
    assert if_any(table_with_data) == True


def test_pyarrow_empty_table():
    empty_table = pa.table({"A": [], "B": []})
    assert if_any(empty_table) == False


# DateTime tests
def test_datetime_objects():
    """All datetime objects should return True."""
    assert if_any(datetime.now()) == True
    assert if_any(date.today()) == True
    assert if_any(time()) == True
    assert if_any(timedelta(days=1)) == True
    assert if_any(timedelta()) == True  # Zero timedelta still returns True


# UUID tests
def test_uuid_objects():
    test_uuid = uuid.uuid4()
    assert if_any(test_uuid) == True

    # Test with specific UUID
    specific_uuid = uuid.UUID("12345678-1234-5678-1234-123456789abc")
    assert if_any(specific_uuid) == True


# PIL Image tests
def test_pil_valid_image():
    valid_image = Image.new("RGB", (10, 10), color="red")
    assert if_any(valid_image) == True


def test_pil_zero_width_image():
    zero_width_image = Image.new("RGB", (0, 10), color="red")
    assert if_any(zero_width_image) == False


def test_pil_zero_height_image():
    zero_height_image = Image.new("RGB", (10, 0), color="red")
    assert if_any(zero_height_image) == False


# File-like objects tests
def test_bytesio_with_content():
    bio_with_content = BytesIO(b"hello world")
    assert if_any(bio_with_content) == True


def test_bytesio_empty():
    empty_bio = BytesIO()
    assert if_any(empty_bio) == False


def test_bytesio_position_preservation():
    """Test that BytesIO position is preserved after if_any call."""
    bio_with_content = BytesIO(b"hello world")
    bio_with_content.read(5)  # Move position
    original_pos = bio_with_content.tell()
    result = if_any(bio_with_content)
    assert result == True
    assert bio_with_content.tell() == original_pos  # Position should be preserved


def test_stringio_with_content():
    sio_with_content = StringIO("hello world")
    assert if_any(sio_with_content) == True


def test_stringio_empty():
    empty_sio = StringIO()
    assert if_any(empty_sio) == False


# Callable tests
def test_function():
    def test_func():
        pass

    assert if_any(test_func) == True


def test_lambda():
    assert if_any(lambda x: x) == True


def test_builtin_function():
    assert if_any(print) == True


# Edge cases tests
def test_file_object_io_error():
    class FileObjectWithIOError:
        def read(self):
            pass

        def seek(self, *args):
            pass

        def tell(self):
            raise IOError("Cannot determine position")

    # Should return True when IOError occurs
    assert if_any(FileObjectWithIOError()) == True


def test_type_objects():
    """Type objects should return True."""
    assert if_any(int) == True
    assert if_any(str) == True
    assert if_any(list) == True


# Integration tests
def test_mixed_data_structures():
    """Test with nested structures."""
    nested_list = [[], [1, 2], {}]
    assert if_any(nested_list) == True

    empty_nested = [[], {}, set()]
    assert if_any(empty_nested) == True  # Non-empty list


def test_real_world_data_scenarios():
    """Test real-world usage patterns."""
    data_samples = [
        pd.DataFrame({"a": [1, 2, 3]}),  # Valid data
        pd.DataFrame(),  # Empty DataFrame
        np.array([1, 2, 3]),  # NumPy array
        "",  # Empty string
        "valid_data",  # Valid string
        None,  # None value
        0,  # Zero (should be True for numbers)
        [],  # Empty list
    ]

    expected = [True, False, True, False, True, False, True, False]

    for data, expected_result in zip(data_samples, expected):
        assert if_any(data) == expected_result


# Parametrized tests for better coverage
@pytest.mark.parametrize(
    "value,expected",
    [
        # Basic types
        (None, False),
        (True, True),
        (False, False),
        (0, True),
        (1, True),
        (-1, True),
        (0.0, True),
        (1.5, True),
        ("", False),
        ("hello", True),
        ([], False),
        ([1], True),
        ({}, False),
        ({"a": 1}, True),
        # Complex numbers
        (complex(0, 0), True),
        (complex(1, 1), True),
        # Bytes
        (b"", False),
        (b"hello", True),
        (bytearray(), False),
        (bytearray(b"test"), True),
    ],
)
def test_if_any_parametrized(value, expected):
    """Parametrized test for common cases."""
    assert if_any(value) == expected
