import os
import pytest
import tempfile
from pathlib import Path
from coded_flows.models import FileMeta, Folder
from coded_flows.utils import FileUtils


# Create a temporary test file for testing
@pytest.fixture(scope="module")
def test_file_path(tmp_path_factory):
    test_file = tmp_path_factory.mktemp("test_files") / "example.txt"
    with open(test_file, "w") as f:
        f.write("Test content\n")
    yield str(test_file)

@pytest.fixture
def test_file_model(test_file_path):
    return FileMeta(filename="example.txt", file_type="text/plain", file_size=100, file_path=test_file_path)

def test_file_model_creation(test_file_model, test_file_path):
    assert test_file_model.filename == "example.txt"
    assert test_file_model.file_type == "text/plain"
    assert test_file_model.file_size == 100
    assert test_file_model.file_path == test_file_path

def test_detect_file_type(test_file_path):
    file_type = FileUtils.detect_file_type(test_file_path)
    assert file_type == "text/plain"

def test_get_file_details(test_file_path):
    file_details = FileUtils.get_file_details(test_file_path)
    assert file_details.filename == os.path.basename(test_file_path)
    assert file_details.file_size == os.path.getsize(test_file_path)
    assert file_details.file_type == "text/plain"
    assert file_details.file_path == test_file_path

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        FileUtils.get_file_details("non_existent_file.txt")

@pytest.fixture
def temp_test_folder():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

def test_folder_model_valid_path(temp_test_folder):
    folder_data = {"folder_path": temp_test_folder}
    folder_model = Folder(**folder_data)
    assert folder_model.folder_path == temp_test_folder

def test_folder_model_invalid_path():
    invalid_folder_path = "/non_existent_folder"
    with pytest.raises(ValueError, match=f"Path '{invalid_folder_path}' is not a directory"):
        folder_data = {"folder_path": invalid_folder_path}
        Folder(**folder_data)

if __name__ == "__main__":
    pytest.main()
