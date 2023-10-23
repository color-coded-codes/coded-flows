from pathlib import Path
from pydantic import BaseModel, field_validator


###################### Files & Folders ######################

class FileMeta(BaseModel):
    file_path: str
    filename: str
    file_type: str
    file_size: int


class Folder(BaseModel):
    folder_path: Path

    @field_validator('folder_path')
    def validate_folder_path(cls, value):
        folder_path = Path(value)
        if not folder_path.is_dir():
            raise ValueError(f"Path '{folder_path}' is not a directory")
        return folder_path

#############################################################