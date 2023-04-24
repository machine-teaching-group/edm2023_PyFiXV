import sys
from typing import Optional, Sequence

sys.path.append(".")


from src.utils.program_utils import parse_python_2, parse_python_3


def is_parseable(generated_repair: str, py_major: int) -> bool:
    if py_major not in {2, 3}:
        raise ValueError(f"py_major cannot be `{py_major}`")

    assert generated_repair is not None, f"generated_repair should not be empty: `{generated_repair}`"
    if py_major == 2:
        return parse_python_2(program=generated_repair)["error_type"] == ""
    elif py_major == 3:
        return parse_python_3(program=generated_repair)["error_type"] == ""


def exact_match(generated_repair: str, groundtruth_repairs: Sequence[str], py_major: int) -> bool:
    """
    Returns: Whether the AST of generated_repair exactly matches at least one of the AST from the groundtruth_repairs
    """
    if not isinstance(groundtruth_repairs, list) and not isinstance(groundtruth_repairs, tuple):
        raise ValueError("groundtruth_repairs needs to be a sequence.")

    if generated_repair is None or not is_parseable(generated_repair=generated_repair, py_major=py_major):
        return False

    if py_major == 2:
        parse_python = parse_python_2
    elif py_major == 3:
        parse_python = parse_python_3
    else:
        raise ValueError("`py_major` must be either 2 or 3")

    generated_source = parse_python(generated_repair)["source"]
    groundtruth_sources = [parse_python(groundtruth)["source"] for groundtruth in groundtruth_repairs]

    return any(generated_source == groundtruth_source for groundtruth_source in groundtruth_sources)


class RepairMetricsClerk:
    """A class for computing all metrics at once"""

    def __init__(self):
        self.n_parseable_fixes = 0
        self.n_exact_match_fixes = 0

    def add(
        self,
        generated_fix: str,
        fixed_programs: Sequence[str],
        buggy_program: str,
        py_major: int,
        parseable_fixes: Optional[Sequence[str]] = None,
    ):
        if generated_fix is not None:
            parseable = True
            is_exact_match = exact_match(generated_fix, fixed_programs, py_major)
        else:
            parseable = False
            is_exact_match = False

        self.n_parseable_fixes += parseable
        self.n_exact_match_fixes += is_exact_match

        info = {
            "generated_fix": generated_fix,
            "is_parseable": parseable,
            "is_exact_match": is_exact_match,
            "parseable_fixes": parseable_fixes,
        }

        return info

    def get_summary(self):
        summary = {
            "n_parseable_fixes": self.n_parseable_fixes,
            "n_exact_match_fixes": self.n_exact_match_fixes,
        }

        return summary
