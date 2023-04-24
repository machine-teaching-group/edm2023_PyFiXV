import sys
from typing import Sequence

sys.path.append(".")


from src.utils.codex_utils import get_codex_edit


def generate_fixes(
    program: str,
    instruction_list: Sequence[str],
    n_list: Sequence[int],
    temperature_list: Sequence[float],
) -> Sequence[str]:
    """
    Use Codex-E to generate multiple fixes.
    """
    assert len(instruction_list) == len(n_list) == len(temperature_list)

    codex_repairs = []
    for instruction, n, temperature in zip(instruction_list, n_list, temperature_list):
        request_output = get_codex_edit(
            input_code=program,
            n=n,
            instruction=instruction,
            temperature=temperature,
        )
        for choice in request_output["choices"]:
            if "text" in choice:
                codex_repairs.append(choice["text"])
            elif "error" in choice:
                print("Codex-E outputs an error")
            else:
                raise ValueError(f"Codex-Edit output is invalid: {choice}")

    return codex_repairs


if __name__ == "__main__":
    pass
