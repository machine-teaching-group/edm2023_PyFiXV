import copy
from pathlib import Path


def serialize_json_value_path(json_data):
    copy_data = copy.deepcopy(json_data)
    for key in copy_data:
        # converse all Path to str
        if isinstance(copy_data[key], Path):
            copy_data[key] = copy_data[key].as_posix()
    return copy_data
