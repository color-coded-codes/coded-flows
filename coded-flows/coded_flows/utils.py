import os
import magic
from .models import FileMeta


class FileUtils:
    @staticmethod
    def detect_file_type(file_path):
        file_type = magic.from_file(file_path, mime=True)
        return file_type

    @staticmethod
    def get_file_details(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError("File not found")

        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_type = FileUtils.detect_file_type(file_path)

        return FileMeta(file_path=file_path, filename=file_name, file_type=file_type, file_size=file_size)
