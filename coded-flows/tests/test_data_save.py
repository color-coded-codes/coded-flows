import pytest
import pandas as pd
import numpy as np
import os
from coded_flows.utils import save_data_to_temp


def test_valid_inputs():
    df = pd.DataFrame({"x": [1, 2, 3], "z": [7, 8, 9]})
    series = pd.Series([100, 200, 300], name="w")
    arr = np.array([20, 21, 22])
    data_records = [{"key1": 30}, {"key1": 31}, {"key1": 32}]

    labels = ["x", "y", "z", "w"]
    file_path = save_data_to_temp(df, series, arr, data_records, labels=labels)

    # Check file existence
    assert os.path.exists(file_path), "Output file does not exist."

    # Check JSON content
    with open(file_path, "r") as f:
        json_content = f.read()

    expected_content = '[{"x":1,"y":100,"z":20,"w":null},{"x":2,"y":200,"z":21,"w":null},{"x":3,"y":300,"z":22,"w":null}]'
    assert json_content.strip() == expected_content, "JSON content mismatch."
    os.remove(file_path)


def test_mismatched_labels():
    df = pd.DataFrame({"x": [1, 2, 3]})
    labels = ["x", "y"]

    with pytest.raises(
        ValueError,
        match="The number of data arguments must match the number of labels.",
    ):
        save_data_to_temp(df, labels=labels)


def test_missing_column_in_dataframe():
    df = pd.DataFrame({"x": [1, 2, 3]})
    labels = ["z"]

    with pytest.raises(ValueError, match="Label 'z' not found in DataFrame columns."):
        save_data_to_temp(df, labels=labels)


def test_invalid_numpy_array():
    arr = np.array([[1, 2], [3, 4]])  # 2D array, invalid
    labels = ["x"]

    with pytest.raises(
        ValueError, match="NumPy array for label 'x' must be one-dimensional."
    ):
        save_data_to_temp(arr, labels=labels)


def test_invalid_data_type():
    invalid_data = {1, 2, 3}  # Set, not supported
    labels = ["x"]

    with pytest.raises(TypeError, match="Unsupported data type: <class 'set'>"):
        save_data_to_temp(invalid_data, labels=labels)


def test_variable_lengths():
    df = pd.DataFrame({"x": [1, 2]})
    arr = np.array([20, 21, 22])  # Different length
    labels = ["x", "y"]

    file_path = save_data_to_temp(df, arr, labels=labels)

    # Check JSON content
    with open(file_path, "r") as f:
        json_content = f.read()

    expected_content = '[{"x":1.0,"y":20},{"x":2.0,"y":21},{"x":null,"y":22}]'
    assert (
        json_content.strip() == expected_content
    ), "JSON content mismatch for variable lengths."
    os.remove(file_path)


def test_series_input():
    series1 = pd.Series([1, 2, 3], name="x")
    series2 = pd.Series([10, 20, 30], name="y")
    labels = ["x", "y"]

    file_path = save_data_to_temp(series1, series2, labels=labels)

    # Check JSON content
    with open(file_path, "r") as f:
        json_content = f.read()

    expected_content = '[{"x":1,"y":10},{"x":2,"y":20},{"x":3,"y":30}]'
    assert (
        json_content.strip() == expected_content
    ), "JSON content mismatch for Series inputs."
    os.remove(file_path)


def test_save_data_to_temp_is_table_true_with_dataframe():
    # Create a sample DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Call the function with `is_table=True`
    json_path = save_data_to_temp(df, is_table=True)

    # Assert the file exists
    assert os.path.exists(json_path), "JSON file was not created."

    # Assert the file content matches the DataFrame content
    with open(json_path, "r") as f:
        saved_data = pd.read_json(f, orient="records")
        pd.testing.assert_frame_equal(saved_data, df)

    # Clean up
    os.remove(json_path)


def test_save_data_to_temp_is_table_true_with_list_of_records():
    # Create a sample list of records
    records = [
        {"col1": 1, "col2": "a"},
        {"col1": 2, "col2": "b"},
        {"col1": 3, "col2": "c"},
    ]

    # Call the function with `is_table=True`
    json_path = save_data_to_temp(records, is_table=True)

    # Assert the file exists
    assert os.path.exists(json_path), "JSON file was not created."

    # Assert the file content matches the list of records
    with open(json_path, "r") as f:
        saved_data = pd.read_json(f, orient="records")
        expected_df = pd.DataFrame.from_records(records)
        pd.testing.assert_frame_equal(saved_data, expected_df)

    # Clean up
    os.remove(json_path)


def test_save_data_to_temp_is_table_true_numpy_array():
    # NumPy array input
    data = np.array([1, 2, 3, 4, 5, 6])
    json_path = save_data_to_temp(data, is_table=True)
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        saved_data = pd.read_json(f, orient="records")
        expected_df = pd.DataFrame(data, columns=["values"])
        pd.testing.assert_frame_equal(saved_data, expected_df)

    # Clean up
    os.remove(json_path)


def test_save_data_to_temp_is_table_true_list():
    # List input
    data = [1, 2, 3, 4, 5, 6]
    json_path = save_data_to_temp(data, is_table=True)
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        saved_data = pd.read_json(f, orient="records")
        expected_df = pd.DataFrame(data, columns=["values"])
        pd.testing.assert_frame_equal(saved_data, expected_df)

    # Clean up
    os.remove(json_path)


def test_save_data_to_temp_is_table_true_tuple():
    # Tuple input
    data = (1, 2, 3, 4, 5, 6)
    json_path = save_data_to_temp(data, is_table=True)
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        saved_data = pd.read_json(f, orient="records")
        expected_df = pd.DataFrame(data, columns=["values"])
        pd.testing.assert_frame_equal(saved_data, expected_df)

    # Clean up
    os.remove(json_path)


def test_save_data_to_temp_is_table_true_series():
    # Series input
    series = pd.Series([1, 2, 3], name="col1")
    json_path = save_data_to_temp(series, is_table=True)
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        saved_data = pd.read_json(f, orient="records")
        expected_df = pd.DataFrame({"values": series})
        pd.testing.assert_frame_equal(saved_data, expected_df)

    # Clean up
    os.remove(json_path)
