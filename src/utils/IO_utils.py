import json
from pathlib import Path
from types import SimpleNamespace

from src.utils.path_utils import serialize_json_value_path


def write_to_file(file_path: Path, content: str, append=False, new_line=True):
    if new_line:
        suffix = "\n"
    else:
        suffix = ""

    with open(file_path, "a" if append else "w") as f:
        f.write(f"{content}{suffix}")


def write_json_file(json_data, file_path: Path):
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=4, sort_keys=True)


def load_json_file(file_path: Path, to_namespace=False):
    with open(file_path, "r") as f:
        if to_namespace:
            content = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        else:
            content = json.load(f)
    return content


def write_argument_file(args, file_path: Path):
    write_json_file(serialize_json_value_path(vars(args)), file_path)
