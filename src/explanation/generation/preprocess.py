import sys
from pathlib import Path
from typing import Sequence

sys.path.append(".")

from src.utils.IO_utils import load_json_file
from src.utils.program_utils import tokenize_and_mask_reliables


def load_annotation_data(
    annotation_data_path: Path,
    partitions: Sequence[str],
):
    """
    Load the annotation data file and:
        - Add `error_text_reliable_masked_tokenized`
    Return: annotation data in JSON format
    """
    assert all(p in {"training", "calibration", "test"} for p in partitions)

    # Load file
    anno_data_json = load_json_file(annotation_data_path)

    # Filter by partitions
    anno_data_json = [instance for instance in anno_data_json if instance["partition"] in partitions]

    # If error_text is None, try to extract error_text from error_line
    for instance in anno_data_json:
        if instance["parser_output"]["error_text"] is None:
            if isinstance(instance["parser_output"]["error_line"], int):
                error_line = instance["parser_output"]["error_line"]
                instance["parser_output"]["error_text"] = instance["buggy_program"].split("\n")[error_line - 1]
            else:
                raise ValueError(f"instance {instance['bad_program_EventID']} has neither error_text nor error_line")

    # Add error_text_reliable_masked_tokenized
    for instance in anno_data_json:
        # get reliable-masked tokenization of the error text
        instance["parser_output"]["error_text_reliable_masked_tokenized"] = tokenize_and_mask_reliables(
            instance["parser_output"]["error_text"]
        )

    # sort by event_id
    anno_data_json = sorted(anno_data_json, key=lambda x: x["bad_program_EventID"])

    return anno_data_json


if __name__ == "__main__":
    pass
