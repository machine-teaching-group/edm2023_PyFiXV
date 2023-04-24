import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.append(".")

from src.utils.IO_utils import load_json_file
from src.utils.program_utils import load_program
from src.validation.utils import split_explanation_to_tokens


def load_explanation_df(
    explanation_grade_path: Path,
    path_to_explanation_folder: Path,
) -> pd.DataFrame:
    """
    Load the explanation-grading dataframe and the explanation folder, make a DataFrame and then return.
    """
    df = pd.DataFrame(load_json_file(explanation_grade_path).items(), columns=["bad_program_EventID", "expert_grade"])
    df["is_correct"] = df["expert_grade"] == 1

    # Codex-generated explanation
    explanation_list = []
    for program_id in df["bad_program_EventID"]:
        explanation_path = path_to_explanation_folder / program_id / "5_prompt_&_explanation.txt"
        explanation_list.append(_extract_explanation_from_file(explanation_path))
    df["explanation"] = explanation_list

    # Buggy program
    df["buggy_program"] = [
        load_program(path_to_explanation_folder / program_id / "1_buggy.py") for program_id in df["bad_program_EventID"]
    ]

    # Fixed program
    df["fixed_program"] = [
        load_program(path_to_explanation_folder / program_id / "2_fixed.py") for program_id in df["bad_program_EventID"]
    ]

    # Length of Codex-generated explanation
    df["explanation_words"] = [len(split_explanation_to_tokens(explanation)) for explanation in df["explanation"]]

    return df


def _extract_explanation_from_file(file_path: Path, nshot: Optional[int] = None) -> str:
    """
    Given a file (whose content include the prompt and the generated explanation),
    extract the generated explanation from that file.
    Optionally assert for the number of few-shot examples with the `nshot` argument.
    """
    explanation_start = "# [FEEDBACK] "
    with open(file_path, "r") as f:
        file_content = f.read()
    if nshot is not None:
        assert file_content.count(explanation_start) == nshot + 1

    explanation = file_content[file_content.rfind(explanation_start) + len(explanation_start) :].strip()
    return explanation
