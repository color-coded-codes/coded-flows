import base64
import json
from typing import Any, Callable, Dict, Tuple, Union
from pydantic_core import MultiHostUrl
import pandas as pd
import pyarrow as pa
from ..types import AnyUrl, Base64Str, Base64Bytes, DataSeries


url_types = [
    "AnyUrl",
    "AnyHttpUrl",
    "HttpUrl",
    "FileUrl",
    "PostgresDsn",
    "CockroachDsn",
    "AmqpDsn",
    "RedisDsn",
    "MongoDsn",
    "KafkaDsn",
    "NatsDsn",
    "MySQLDsn",
    "MariaDBDsn",
]

json_value_types = [
    "List",
    "Dict",
    "Str",
    "Base64Str",
    "CountryAlpha2",
    "CountryAlpha3",
    "CountryNumericCode",
    "CountryShortName",
    "EmailStr",
    "Currency",
    "MacAddress",
    "Bool",
    "Int",
    "Float",
    "Null",
]

core_string_types = ["Str", "AnyStr"]

string_types = [
    "Str",
    "Base64Str",
    "CountryAlpha2",
    "CountryAlpha3",
    "CountryNumericCode",
    "CountryShortName",
    "EmailStr",
    "Currency",
    "Json",
    "MacAddress",
]


conversion_mapping = {
    "Any": [],
    "Null": ["Json"],
    # Data
    "DataSeries": [
        "DataFrame",
        "List",
        "Tuple",
        "Set",
        "Json",
        "NDArray",
        "ArrowTable",
    ],  # <-- works as a Helper
    "DataFrame": [
        "DataRecords",
        "List",
        "Dict",
        "Json",
        "NDArray",
        "ArrowTable",
        "DataDict",
    ],
    "ArrowTable": [
        "DataFrame",
        "DataRecords",
        "List",
        "Dict",
        "Json",
        "NDArray",
        "DataDict",
    ],
    "NDArray": ["List", "Json"],
    "DataDict": [
        "DataFrame",
        "ArrowTable",
    ],
    "DataRecords": [
        "DataFrame",
        "ArrowTable",
    ],
    # Strings
    "Str": ["Json", "Base64Str", "Base64Bytes"],
    "AnyStr": [],
    "Base64Str": ["Json", "Base64Bytes"],
    # Country - str too
    "CountryAlpha2": ["Json", "Base64Str", "Base64Bytes"],  # <-- works as a Helper
    "CountryAlpha3": ["Json", "Base64Str", "Base64Bytes"],  # <-- works as a Helper
    "CountryNumericCode": ["Json", "Base64Str", "Base64Bytes"],  # <-- works as a Helper
    "CountryShortName": ["Json", "Base64Str", "Base64Bytes"],  # <-- works as a Helper
    # Currency - str too
    "Currency": ["Json", "Base64Str", "Base64Bytes"],
    # MacAddress - str too
    "MacAddress": ["Json", "Base64Str", "Base64Bytes"],
    # Email - str too
    "EmailStr": ["Json", "Base64Str", "Base64Bytes"],
    # Boolean
    "Bool": ["Json"],
    # Datetime
    "Datetime": [],  # <-- works as a Helper
    "Date": [],  # <-- works as a Helper
    "Time": [],  # <-- works as a Helper
    "Timedelta": [],  # <-- works as a Helper
    # Numbers
    "Int": ["Json"],
    "Float": ["Json"],
    "Complex": [],
    "Decimal": [],
    "Number": [],
    "PositiveInt": [],
    "NegativeInt": [],
    "PositiveFloat": [],
    "NegativeFloat": [],
    "FiniteFloat": [],
    "ByteSize": [],  # <-- works as a Helper
    # Iterables
    "List": ["Json"],
    "Tuple": [],
    "Deque": [],
    "Set": [],
    "FrozenSet": [],
    "Iterable": [],
    # Mapping
    "Dict": ["Json"],
    # Callable
    "Callable": [],
    # IP Address types
    "IPvAnyAddress": [],  # <-- works as a Helper
    "IPvAnyInterface": [],  # <-- works as a Helper
    "IPvAnyNetwork": [],  # <-- works as a Helper
    # network types
    "AnyUrl": [],  # <-- works as a Helper
    "AnyHttpUrl": [],  # <-- works as a Helper
    "HttpUrl": [],  # <-- works as a Helper
    "FileUrl": [],  # <-- works as a Helper
    "PostgresDsn": [],  # <-- works as a Helper
    "CockroachDsn": [],  # <-- works as a Helper
    "AmqpDsn": [],  # <-- works as a Helper
    "RedisDsn": [],  # <-- works as a Helper
    "MongoDsn": [],  # <-- works as a Helper
    "KafkaDsn": [],  # <-- works as a Helper
    "NatsDsn": [],  # <-- works as a Helper
    "MySQLDsn": [],  # <-- works as a Helper
    "MariaDBDsn": [],  # <-- works as a Helper
    # bytes
    "Bytes": [],
    "Bytearray": [],
    "Base64Bytes": ["Base64Str"],
    "BytesIOType": [],
    # Paths
    "Path": [],  # <-- works as a Helper
    "NewPath": [],  # <-- works as a Helper
    "FilePath": [],  # <-- works as a Helper
    "DirectoryPath": [],  # <-- works as a Helper
    # UUID
    "UUID": [],
    "UUID1": [],
    "UUID3": [],
    "UUID4": [],
    "UUID5": [],
    # Json
    "JsonValue": [],
    "Json": ["Base64Str", "Base64Bytes"],
    # Secret
    "SecretStr": [],  # <-- works as a Helper
    # Color
    "Color": [],  # <-- works as a Helper
    # Coordinates
    "Longitude": [],
    "Latitude": [],
    "Coordinate": [],  # <-- works as a Helper
    # Media
    "PILImage": [],
    "MediaData": [],
}


def dataseries_to_type(output_type: str) -> Callable:
    if output_type == "DataFrame":
        return lambda x: x.to_frame()
    elif output_type == "List":
        return lambda x: x.to_list()
    elif output_type == "Tuple":
        return lambda x: tuple(x.to_list())
    elif output_type == "Set":
        return lambda x: set(x.to_list())
    elif output_type == "Json":
        return lambda x: x.to_json()
    elif output_type == "NDArray":
        return lambda x: x.to_numpy()
    elif output_type == "ArrowTable":
        return lambda x: pa.Table.from_pandas(x.to_frame())


def dataframe_to_type(output_type: str) -> Callable:
    if output_type == "DataRecords":
        return lambda x: x.to_dict("records")
    elif output_type == "List":
        return lambda x: x.to_dict("records")
    elif output_type == "Dict":
        return lambda x: x.to_dict("list")
    elif output_type == "DataDict":
        return lambda x: x.to_dict("list")
    elif output_type == "Json":
        return lambda x: x.to_json(orient="records")
    elif output_type == "NDArray":
        return lambda x: x.to_numpy()
    elif output_type == "ArrowTable":
        return lambda x: pa.Table.from_pandas(x)


def arrow_to_type(output_type: str) -> Callable:
    if output_type == "DataRecords":
        return lambda x: x.to_pylist()
    elif output_type == "List":
        return lambda x: x.to_pylist()
    elif output_type == "Dict":
        return lambda x: x.to_pydict()
    elif output_type == "DataDict":
        return lambda x: x.to_pydict()
    elif output_type == "Json":
        return lambda x: json.dumps(x.to_pylist())
    elif output_type == "NDArray":
        return lambda x: x.to_pandas().to_numpy()
    elif output_type == "DataFrame":
        return lambda x: x.to_pandas()


def numpy_to_type(output_type: str) -> Callable:
    if output_type == "List":
        return lambda x: x.tolist()
    elif output_type == "Json":
        return lambda x: json.dumps(x.tolist())


def datadict_to_type(output_type: str) -> Callable:
    if output_type == "DataFrame":
        return lambda x: pd.DataFrame(x)
    elif output_type == "ArrowTable":
        return lambda x: pa.Table.from_pydict(x)


def datarecords_to_type(output_type: str) -> Callable:
    if output_type == "DataFrame":
        return lambda x: pd.DataFrame(x)
    elif output_type == "ArrowTable":
        return lambda x: pa.Table.from_pylist(x)


def jsonify(value: Any) -> str:
    return json.dumps(value, skipkeys=True)


def str_to_bytes(input_string: str) -> bytes:
    return input_string.encode("utf-8")


def base64str_to_base64bytes(base64_string: str) -> bytes:
    return str_to_bytes(base64_string)


def url_to_str(value: Union[AnyUrl, MultiHostUrl, str]) -> str:
    if isinstance(value, str):
        return value
    return value.unicode_string()


def url_to_bytes(input_string: str) -> bytes:
    url_txt = url_to_str(input_string)
    return str_to_bytes(url_txt)


def url_to_base64str(value: Union[AnyUrl, MultiHostUrl, str]) -> Base64Str:
    url_txt = url_to_str(value)
    url_txt_bytes = url_txt.encode("utf-8")
    base64_bytes = base64.b64encode(url_txt_bytes)
    base64_string = base64_bytes.decode("utf-8")
    return base64_string


def base64bytes_to_base64str(value: Base64Bytes) -> Base64Str:
    base64_string = value.decode("utf-8")
    return base64_string


def url_to_base64bytes(value: Union[AnyUrl, MultiHostUrl, str]) -> Base64Bytes:
    url_txt = url_to_str(value)
    url_txt_bytes = url_txt.encode("utf-8")
    base64_bytes = base64.b64encode(url_txt_bytes)
    return base64_bytes


def str_to_base64str(value: str) -> Base64Str:
    value_bytes = value.encode("utf-8")
    base64_bytes = base64.b64encode(value_bytes)
    base64_string = base64_bytes.decode("utf-8")
    return base64_string


def str_to_base64bytes(value: str) -> Base64Str:
    value_bytes = value.encode("utf-8")
    base64_bytes = base64.b64encode(value_bytes)
    return base64_bytes


def get_conversion_function(input_type: str, output_type: str) -> Callable:
    if input_type in json_value_types and output_type == "Json":
        return jsonify
    elif input_type == "DataSeries" and output_type in conversion_mapping["DataSeries"]:
        return dataseries_to_type(output_type)
    elif input_type == "DataFrame" and output_type in conversion_mapping["DataFrame"]:
        return dataframe_to_type(output_type)
    elif input_type == "ArrowTable" and output_type in conversion_mapping["ArrowTable"]:
        return arrow_to_type(output_type)
    elif input_type == "NDArray" and output_type in conversion_mapping["NDArray"]:
        return numpy_to_type(output_type)
    elif input_type == "DataDict" and output_type in conversion_mapping["DataDict"]:
        return datadict_to_type(output_type)
    elif (
        input_type == "DataRecords" and output_type in conversion_mapping["DataRecords"]
    ):
        return datarecords_to_type(output_type)
    elif (
        input_type in [t for t in string_types if t != "Base64Str"]
        and output_type == "Base64Str"
    ):
        return str_to_base64str
    elif input_type in string_types and output_type == "Bytes":
        return str_to_bytes
    elif (
        input_type in [t for t in string_types if t != "Base64Str"]
        and output_type == "Base64Bytes"
    ):
        return str_to_base64bytes
    elif input_type == "Base64Str" and output_type == "Base64Bytes":
        return base64str_to_base64bytes
    elif input_type == "Base64Bytes" and output_type == "Base64Str":
        return base64bytes_to_base64str
    return None


def convert_type(value: any, input_type: str, output_type: str) -> any:
    conversion_function = get_conversion_function(input_type, output_type)

    if conversion_function:
        return conversion_function(value)

    return value
