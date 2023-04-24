import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.append(".")

from src.explanation.generation.select_fewshot import ExplanationFewShotExample
from src.utils.IO_utils import load_json_file, write_json_file, write_to_file
from src.utils.program_utils import get_program_diff, get_program_token_diff


def load_cached_repairs(cached_repair_path: Path, k: int, event_id: str):
    """
    Load cached repairs.
    This is to increase consistency when comparing explanation methods, and save time as we do not need to
        generate repairs any time we want to try a explanation method.
    """
    instance_path = cached_repair_path / str(k) / event_id
    cached_metrics_path = instance_path / "instance_combined_metrics.json"
    cached_results_summary_path = cached_repair_path / "results.json"

    cached_metrics = load_json_file(cached_metrics_path)
    cached_fix = cached_metrics["generated_fix"]
    cached_results_summary = load_json_file(cached_results_summary_path)["results_summary"][str(k)]

    return {
        "generated_fix": cached_fix,
        "instance_combined_metrics": cached_metrics,
        "results_summary": cached_results_summary,
    }


def write_explanation_instance_result(
    output_path: Path,
    event_id: str,
    buggy_program: str,
    generated_fix: Optional[str],
    prompt: Optional[str],
    generated_explanation: Optional[str],
    few_shots: Sequence[ExplanationFewShotExample],
):
    if output_path is not None:
        if generated_explanation is not None:
            explanation_given_path = output_path / "explanation_given"
            explanation_given_path.mkdir(exist_ok=True)
            event_output_path = explanation_given_path / event_id
        else:
            explanation_none_path = output_path / "explanation_none"
            explanation_none_path.mkdir(exist_ok=True)
            event_output_path = explanation_none_path / event_id

        event_output_path.mkdir()
        write_to_file(event_output_path / "1_buggy.py", buggy_program)
        write_to_file(event_output_path / "2_fixed.py", generated_fix)

        if generated_fix is not None:
            few_shot_criteria = []
            for ishot, shot in enumerate(few_shots):
                shot_crit = {
                    "name": f"shot_{ishot}",
                    # "event_id": shot.bad_program_EventID,
                    "token_diff": get_program_token_diff(shot.buggy_program, shot.fixed_program),
                }
                few_shot_criteria.append(shot_crit)
            few_shot_criteria.append(
                {"name": "testing_example", "token_diff": get_program_token_diff(buggy_program, generated_fix)}
            )
            write_json_file(few_shot_criteria, event_output_path / "3_few_shot_criteria.json")

            write_to_file(
                event_output_path / "4_few_shot_explanation.txt",
                "\n\n[SEPARATION]\n\n".join(shot.explanation for shot in few_shots),
            )

            write_to_file(
                event_output_path / "5_prompt_&_explanation.txt", str(prompt) + " " + str(generated_explanation)
            )

            diff = get_program_diff(buggy_program, generated_fix)
            write_to_file(
                event_output_path / "6_buggy_&_diff_&_explanation.txt",
                f"{buggy_program}\n\n=== [DIFF] ===\n{diff}\n=== [FEEDBACK] ===\n{generated_explanation}",
            )
