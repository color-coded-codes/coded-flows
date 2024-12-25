import os
import pytest
from PIL import Image
import numpy as np
from io import BytesIO
from coded_flows.utils import save_image_to_temp


# ============ Fixtures ============
@pytest.fixture
def sample_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_ndarray() -> np.ndarray:
    # Create a 100x100 red image as a numpy array
    return np.zeros((100, 100, 3)) + [255, 0, 0]


@pytest.fixture
def sample_bytes_io() -> BytesIO:
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)  # Reset buffer pointer
    return buffer


@pytest.fixture
def sample_pil_image() -> Image.Image:
    return Image.new("RGB", (100, 100), color="green")


# ============== Tests =============


def test_save_image_from_bytes(sample_bytes):
    """Test saving an image from bytes."""
    file_path = save_image_to_temp(sample_bytes)
    assert os.path.exists(file_path), "File was not created"
    assert file_path.endswith(".png"), "File does not have the correct extension"

    saved_img = Image.open(file_path)
    assert saved_img.size == (100, 100), "Resolution does not match the input image"

    os.remove(file_path)  # Clean up


def test_save_image_from_ndarray(sample_ndarray):
    """Test saving an image from a numpy ndarray."""
    file_path = save_image_to_temp(sample_ndarray)
    assert os.path.exists(file_path), "File was not created"
    assert file_path.endswith(".png"), "File does not have the correct extension"

    saved_img = Image.open(file_path)
    assert saved_img.size == (100, 100), "Resolution does not match the input image"

    os.remove(file_path)  # Clean up


def test_save_image_from_bytes_io(sample_bytes_io):
    """Test saving an image from BytesIO."""
    file_path = save_image_to_temp(sample_bytes_io)
    assert os.path.exists(file_path), "File was not created"
    assert file_path.endswith(".png"), "File does not have the correct extension"

    saved_img = Image.open(file_path)
    assert saved_img.size == (100, 100), "Resolution does not match the input image"

    os.remove(file_path)  # Clean up


def test_save_image_from_pil_image(sample_pil_image):
    """Test saving an image from a PIL Image."""
    file_path = save_image_to_temp(sample_pil_image)
    assert os.path.exists(file_path), "File was not created"
    assert file_path.endswith(".png"), "File does not have the correct extension"

    saved_img = Image.open(file_path)
    assert saved_img.size == (100, 100), "Resolution does not match the input image"

    os.remove(file_path)  # Clean up


def test_save_jpeg_image_as_png():
    """Test saving a JPEG image as PNG."""
    img = Image.new("RGB", (100, 100), color="yellow")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    jpeg_image = Image.open(buffer)

    file_path = save_image_to_temp(jpeg_image)
    assert os.path.exists(file_path), "File was not created"
    assert file_path.endswith(".png"), "JPEG image was not converted to PNG"

    saved_img = Image.open(file_path)
    assert saved_img.format == "PNG", "Saved image is not in PNG format"
    assert saved_img.size == (100, 100), "Resolution does not match the input image"

    os.remove(file_path)  # Clean up
