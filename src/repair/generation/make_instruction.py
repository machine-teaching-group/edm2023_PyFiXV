import sys

sys.path.append(".")

from src.utils.program_utils import get_parsing_error_message


def make_repair_instruction(instruction_type: str, py_major: int, config: dict, program_path=None) -> str:
    """
    Return an instruction for Codex-Edit repair.
    """
    if instruction_type == "general":
        instruction = config["Codex_e"]["instruction"]["general"].format(py_major=py_major)
    elif instruction_type == "specific":
        if py_major == 2:
            raise ValueError(f"`instruction` == {instruction_type} is not supported for Python 2")
        elif py_major == 3:
            if program_path is not None:
                error = get_parsing_error_message(program_path)
                instruction = config["Codex_e"]["instruction"]["specific"].format(
                    py_major=py_major,
                    err_mes=error.msg.replace("\n", " "),
                    line_content=error.text,
                )
            else:
                raise ValueError("To generate specific instruction for Python 3, program_path is required")
        else:
            raise ValueError(f"Python version must be either 2 or 3, not {py_major}")
    else:
        raise ValueError(f"No support for instruction type {instruction_type}")

    return instruction


if __name__ == "__main__":
    pass
