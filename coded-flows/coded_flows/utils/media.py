import os
import uuid
import tempfile
from typing import Union, List, Dict, Any
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image


def save_image_to_temp(image):
    random_filename = f"cfimg_{uuid.uuid4().hex}.png"
    temp_dir = os.path.join(tempfile.gettempdir(), "coded-flows-media")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, random_filename)

    try:

        if isinstance(image, bytes):
            image = BytesIO(image)

        if isinstance(image, BytesIO):
            image = Image.open(image)

        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        if isinstance(image, Image.Image):
            image.save(file_path, format="PNG")
        else:
            raise ValueError("Unsupported image type provided.")

    except Exception as e:
        raise ValueError(f"Failed to save image: {e}")

    return file_path


# List, DataSeries, NDArray, DataRecords, DataFrame
def save_data_to_temp(
    *data_args: Union[
        pd.DataFrame, pd.Series, np.ndarray, List[Dict[str, Any]], List[Any]
    ],
    labels: List[str],
) -> str:
    if len(data_args) != len(labels):
        raise ValueError(
            "The number of data arguments must match the number of labels."
        )

    normalized_data = []
    max_length = 0

    for data, label in zip(data_args, labels):
        if isinstance(data, pd.DataFrame):
            if label not in data.columns:
                raise ValueError(f"Label '{label}' not found in DataFrame columns.")
            col_data = data[label].values
        elif isinstance(data, pd.Series):
            col_data = data.values
        elif isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ValueError(
                    f"NumPy array for label '{label}' must be one-dimensional."
                )
            col_data = data
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            col_data = [item.get(label, None) for item in data]
        elif isinstance(data, list):
            col_data = data
        elif isinstance(data, tuple):
            col_data = list(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        normalized_data.append(pd.Series(col_data, name=label))
        max_length = max(max_length, len(col_data))

    combined_df = pd.concat(normalized_data, axis=1)

    combined_df = combined_df.reindex(range(max_length)).reset_index(drop=True)

    random_filename = f"cfdata_{uuid.uuid4().hex}.json"
    temp_dir = os.path.join(tempfile.gettempdir(), "coded-flows-media")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, random_filename)
    combined_df.to_json(file_path, orient="records", lines=False, indent=None)

    return file_path
