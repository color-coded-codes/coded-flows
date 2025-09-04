import os
import tempfile
import uuid
import pytest
from coded_flows.utils.media import save_text_to_temp


def test_save_text_to_temp_str(tmp_path, monkeypatch):
    """Test saving a simple string input."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    text = "hello world"
    path = save_text_to_temp(text, "testfile")

    # Ensure correct path
    assert path.endswith("coded-flows-media/cfdata_testfile.txt")
    assert os.path.exists(path)

    # Ensure correct content
    with open(path, "r", encoding="utf-8") as f:
        assert f.read() == text


def test_save_text_to_temp_bytes(monkeypatch, tmp_path):
    """Test saving UTF-8 encodable bytes input."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    data = "hello bytes".encode("utf-8")
    path = save_text_to_temp(data, "bytesfile")

    with open(path, "r", encoding="utf-8") as f:
        assert f.read() == "hello bytes"


def test_save_text_to_temp_bytes_not_decodable(monkeypatch, tmp_path):
    """Test saving non-UTF-8 bytes raises TypeError."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    data = b"\xff\xfe"  # invalid UTF-8

    with pytest.raises(TypeError, match="not decodable"):
        save_text_to_temp(data, "badbytes")


def test_save_text_to_temp_invalid_type(monkeypatch, tmp_path):
    """Test passing an unsupported type raises TypeError."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    with pytest.raises(TypeError, match="Expected str or bytes"):
        save_text_to_temp(1234, "invalid")


def test_save_text_to_temp_random_filename(monkeypatch, tmp_path):
    """Test that UUID-based filenames are generated when filename is None."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    fake_uuid = uuid.UUID("12345678123456781234567812345678")
    monkeypatch.setattr(uuid, "uuid4", lambda: fake_uuid)

    path = save_text_to_temp("random", None)
    assert path.endswith(
        "coded-flows-media/cfdata_12345678123456781234567812345678.txt"
    )
    assert os.path.exists(path)


def test_directory_creation(monkeypatch, tmp_path):
    """Ensure directory is created if not existing."""
    target_dir = tmp_path / "nested-temp"
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(target_dir))

    path = save_text_to_temp("make dir", "dirtest")
    assert os.path.exists(os.path.dirname(path))
