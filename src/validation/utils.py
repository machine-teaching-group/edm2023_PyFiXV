import re
import sys
from pathlib import Path
from typing import Sequence

sys.path.append(".")

from src.utils.IO_utils import write_to_file


def split_explanation_to_tokens(explanation: str) -> Sequence[str]:
    re_split_pattern = r", |\. |\? |! |\.$| |\b"  # split by `, ` or `. ` or `? ` or `! ` or `.` or ` ` or boundary
    tokens = [each for each in re.split(re_split_pattern, explanation) if each]
    return tokens


def write_validation_instance_result(
    output_path: Path,
    buggy_program: str,
    target_fix: str = None,
    validation_instruction: str = None,
    validation_result: str = None,
    explanation_length: int = None,
):
    write_to_file(output_path / "buggy_program.py", buggy_program)
    if target_fix:
        write_to_file(output_path / "target_fix.py", target_fix)
    if validation_instruction:
        write_to_file(output_path / "validation_instruction.txt", validation_instruction)
    if validation_result:
        write_to_file(output_path / "validation_result.txt", validation_result)
    if explanation_length:
        write_to_file(output_path / "explanation_length.txt", str(explanation_length))
