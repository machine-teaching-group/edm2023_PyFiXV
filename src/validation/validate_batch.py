import argparse
import sys
from collections import abc
from pathlib import Path
from typing import Sequence

import pandas as pd
from tqdm import tqdm

sys.path.append(".")

from src.utils.IO_utils import write_json_file
from src.utils.path_utils import serialize_json_value_path
from src.validation.methods import verify_by_back_repair_multiple_thresholds
from src.validation.preprocess import load_explanation_df
from src.validation.utils import write_validation_instance_result


def check_and_parse_arguments(args):
    if not args.config_path.exists():
        raise ValueError(f"config_path `{args.config_path}` does not exist.")

    if not args.existing_explanation_folder.exists():
        raise ValueError(f"existing_explanation_folder {args.existing_explanation_folder} does not exist.")

    if not args.existing_human_grade_path.exists():
        raise ValueError(f"existing_human_grade_path {args.existing_human_grade_path} does not exist.")

    if args.py_major not in {2, 3}:
        raise ValueError(f"py_major must be either 2 or 3 (current value is {args.py_major}).")

    if args.verify_by_back_repair_exact_match + args.verify_by_explanation_length != 1:
        raise ValueError("Please specify exactly 1 validation method.")

    args.back_repair_n_list = eval(args.back_repair_n_list)
    if not isinstance(args.back_repair_n_list, abc.Sequence):
        raise ValueError("`back_repair_n_list` must be a sequence of integers")
    for each in args.back_repair_n_list:
        if not isinstance(each, int):
            raise ValueError("`back_repair_n_list` must be a sequence of integers")

    args.back_repair_temperature_list = eval(args.back_repair_temperature_list)
    if not isinstance(args.back_repair_temperature_list, abc.Sequence):
        raise ValueError("`back_repair_temperature_list` must be a sequence of float")
    for each in args.back_repair_temperature_list:
        if not (isinstance(each, float) or isinstance(each, int)):
            raise ValueError("`back_repair_temperature_list` must be a sequence of floats")

    if len(args.back_repair_n_list) != len(args.back_repair_temperature_list):
        raise ValueError(
            f"`back_repair_n_list` ({args.back_repair_n_list}) and `back_repair_temperature_list` "
            f"({args.back_repair_temperature_list}) must have the same length"
        )

    if not args.output_path.parents[0].exists():
        raise ValueError(f"output folder `{args.output_path.parents[0]}` does not exist.")
    if args.output_path.exists() and any(args.output_path.iterdir()):
        raise ValueError(f"output_path {args.output_path} already existed and is non-empty.")
    args.output_path.mkdir(exist_ok=True)

    return args


def validate_by_back_repair(args, explanation_df: pd.DataFrame):
    # Initialize
    output_all = {"arguments": serialize_json_value_path(vars(args)), "results": {}, "result_summary": {}}
    (args.output_path / "programs").mkdir(exist_ok=True)

    # Verify by Back-Repair
    back_repair_pass_thresholds = list(range(1, sum(args.back_repair_n_list) + 1))
    for _, row in tqdm(explanation_df.iterrows()):
        event_id = row["bad_program_EventID"]
        instance_verify_result = [True] * len(back_repair_pass_thresholds)

        instruction = row["explanation"]
        verify_result = [
            result.ExactMatch
            for result in verify_by_back_repair_multiple_thresholds(
                buggy_program=row["buggy_program"],
                fixed_program=row["fixed_program"],
                generated_explanation=instruction,
                py_major=args.py_major,
                n_list=args.back_repair_n_list,
                temperature_list=args.back_repair_temperature_list,
                pass_thresholds=back_repair_pass_thresholds,
            )
        ]
        instance_verify_result = [x & y for x, y in zip(instance_verify_result, verify_result)]

        # Write output of this program
        ofolder = args.output_path / "programs" / event_id
        ofolder.mkdir()
        write_validation_instance_result(
            output_path=ofolder,
            buggy_program=row["buggy_program"],
            target_fix=row["fixed_program"],
            validation_instruction=instruction,
            validation_result=f"{sum(instance_verify_result)}/{len(instance_verify_result)} match(es).",
        )

        output_all["results"][event_id] = {
            "pass_thresholds": {
                pass_threshold: verification
                for pass_threshold, verification in zip(back_repair_pass_thresholds, instance_verify_result)
            }
        }

    for pass_threshold in back_repair_pass_thresholds:
        accepted_event_id_list = []
        for event_id in output_all["results"].keys():
            if output_all["results"][event_id]["pass_thresholds"][pass_threshold]:
                accepted_event_id_list.append(event_id)
        accepted_df = explanation_df[explanation_df["bad_program_EventID"].isin(accepted_event_id_list)]
        precision = accepted_df["is_correct"].mean() if len(accepted_df) else None
        coverage = len(accepted_df) / len(explanation_df)

        output_all["result_summary"][pass_threshold] = {
            "precision": precision,
            "coverage": coverage,
        }

    # Write output to file
    write_json_file(output_all, args.output_path / "results.json")
    print(f"Write results to {args.output_path}")


def validate_by_explanation_length(
    args,
    explanation_df: pd.DataFrame,
    length_thresholds: Sequence[int] = range(30, 201, 10),
):
    # Initialize
    output_all = {"arguments": serialize_json_value_path(vars(args)), "results": {}, "result_summary": {}}
    (args.output_path / "programs").mkdir(exist_ok=True)

    # Verify by explanation length
    for _, row in tqdm(explanation_df.iterrows()):
        event_id = row["bad_program_EventID"]
        explanation_length = row["explanation_words"]

        output_all["results"][event_id] = {
            "length_thresholds": {threshold: explanation_length < threshold for threshold in length_thresholds}
        }

        # Write output of this program
        ofolder = args.output_path / "programs" / event_id
        ofolder.mkdir()
        write_validation_instance_result(
            output_path=ofolder,
            buggy_program=row["buggy_program"],
            explanation_length=explanation_length,
        )

    # Compute over-all precision and coverage
    for threshold in length_thresholds:
        covering = 0
        success_count = 0
        for _, row in explanation_df.iterrows():
            explanation_length = row["explanation_words"]
            if explanation_length < threshold:
                covering += 1
                if row["is_correct"]:
                    success_count += 1

        coverage = covering / len(explanation_df)
        precision = success_count / covering if covering else 0
        output_all["result_summary"][threshold] = {
            "precision": precision,
            "coverage": coverage,
        }

    # Write output to file
    write_json_file(output_all, args.output_path / "results.json")
    print(f"Write results to {args.output_path}")


def main(args):
    explanation_df = load_explanation_df(
        explanation_grade_path=args.existing_human_grade_path,
        path_to_explanation_folder=args.existing_explanation_folder,
    )
    print(explanation_df.head())

    if args.verify_by_back_repair_exact_match:
        validate_by_back_repair(args, explanation_df=explanation_df)
    elif args.verify_by_explanation_length:
        validate_by_explanation_length(args, explanation_df=explanation_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=Path, nargs="?", default="config.json", help="Path to the JSON config file."
    )
    parser.add_argument("--existing_explanation_folder", type=Path, help="Path to the existing explanation folder.")
    parser.add_argument(
        "--existing_human_grade_path", type=Path, help="Path to the JSON human-grade of the explanation."
    )
    parser.add_argument("--py_major", type=int, help="Python major version of the programs.")
    parser.add_argument(
        "--verify_by_back_repair_exact_match",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Validate feedback by checking if back-repairs using explanation results in exact-matches.",
    )
    parser.add_argument(
        "instruction_type", type=str, nargs="?", default="general", help="Type of instruction for Codex-E."
    )
    parser.add_argument(
        "--verify_by_explanation_length",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Validate feedback by checking if the length of explanation is shorter than some thresholds.",
    )
    parser.add_argument(
        "--back_repair_n_list",
        type=str,
        nargs="?",
        default="[1]",
        help="List of `n` for generating back-repairs.",
    )
    parser.add_argument(
        "--back_repair_temperature_list",
        type=str,
        nargs="?",
        default="[0.]",
        help="List of `temperature` for generating back-repairs.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to the output folder.",
    )

    args = parser.parse_args()
    args = check_and_parse_arguments(args)
    main(args)
