import argparse
import random
import sys
from collections import abc
from itertools import accumulate
from pathlib import Path

from tqdm import tqdm

sys.path.append(".")

from src.end_to_end.evaluation.metrics import End2EndClerk
from src.explanation.generation.preprocess import load_annotation_data
from src.repair.evaluation.metrics import RepairMetricsClerk
from src.repair.generation.generate_fixes import generate_fixes
from src.repair.generation.make_instruction import make_repair_instruction
from src.repair.generation.preprocess import strip_program
from src.repair.generation.select_best_fix import select_fix_by_ed
from src.utils.IO_utils import load_json_file, write_json_file, write_to_file
from src.utils.path_utils import serialize_json_value_path
from src.utils.program_utils import get_program_diff

random.seed(42)


def verify_arguments(args):
    if not args.config_path.exists():
        raise ValueError(f"config_path `{args.config_path}` does not exist.")

    if not args.annotation_data_path.exists():
        raise ValueError(f"annotation_data_path `{args.annotation_data_path}` does not exist.")

    args.partitions = args.partitions.split(",")

    if args.instruction_type not in {"general", "specific"}:
        raise ValueError(f"instruction_type `{args.instruction_type} is not valid.")

    args.n_list = eval(args.n_list)
    if not isinstance(args.n_list, abc.Sequence):
        raise ValueError("`n_list` must be a sequence of integers")
    for each in args.n_list:
        if not isinstance(each, int):
            raise ValueError("`n_list` must be a sequence of integers")

    args.temperature_list = eval(args.temperature_list)
    if not isinstance(args.temperature_list, abc.Sequence):
        raise ValueError("`temperature_list` must be a sequence of floats")
    for each in args.temperature_list:
        if not (isinstance(each, float) or isinstance(each, int)):
            raise ValueError("`temperature_list` must be a sequence of floats")

    if len(args.n_list) != len(args.temperature_list):
        raise ValueError(
            f"`n_list` ({args.n_list}) and `temperature_list` " f"({args.temperature_list}) must have the same length"
        )

    if args.output_path is not None:
        if not args.output_path.parents[0].exists():
            raise ValueError(f"output folder `{args.output_path.parents[0]}` does not exist.")
        if args.output_path.exists() and any(args.output_path.iterdir()):
            raise ValueError(f"output_path {args.output_path} already existed and is non-empty.")
    args.output_path.mkdir(exist_ok=True)


def main(args):
    config = load_json_file(args.config_path)
    anno_data = load_annotation_data(annotation_data_path=args.annotation_data_path, partitions=args.partitions)

    repair_clerks = [RepairMetricsClerk() for _ in range(len(args.n_list))]
    end2end_clerks = [End2EndClerk() for _ in range(len(args.n_list))]
    output_all = {"arguments": serialize_json_value_path(vars(args)), "results": {}, "results_summary": {}}
    for example in (pbar := tqdm(anno_data)):
        [eclerk.start() for eclerk in end2end_clerks]
        # Get required information
        event_id = example["bad_program_EventID"]
        buggy_program = strip_program(example["buggy_program"])
        py_major = example["python_major"]
        fixed_programs = [anno["fixed_program"] for anno in example["annotation"]]
        instruction = make_repair_instruction(instruction_type=args.instruction_type, py_major=py_major, config=config)

        # Compute the fixes
        all_generated_fixes = []
        for i, (n, temperature, repair_clerk, end2end_clerk, total_n) in enumerate(
            zip(args.n_list, args.temperature_list, repair_clerks, end2end_clerks, accumulate(args.n_list))
        ):
            # get and append the new generated fixes
            [eclerk.unpause() for eclerk in end2end_clerks[i:]]
            generated_fixes = generate_fixes(
                program=buggy_program, instruction_list=[instruction], n_list=[n], temperature_list=[temperature]
            )
            all_generated_fixes += generated_fixes
            [eclerk.unpause() for eclerk in end2end_clerks[i + 1 :]]

            # select the best fix
            best_fix, _, parseable_fixes = select_fix_by_ed(
                buggy_program=buggy_program,
                generated_fixes=all_generated_fixes,
                py_major=py_major,
            )

            # Compute metrics
            instance_repair_eval = end2end_clerk.end()
            instance_repair_metrics = repair_clerk.add(
                generated_fix=best_fix,
                parseable_fixes=parseable_fixes,
                fixed_programs=fixed_programs,
                buggy_program=buggy_program,
                py_major=py_major,
            )
            instance_combined_metrics = instance_repair_metrics | instance_repair_eval

            if total_n not in output_all["results"]:
                output_all["results"][total_n] = {}
            output_all["results"][total_n][event_id] = instance_combined_metrics

            # Write fixed program, evaluation metrics to files
            if args.output_path is not None:
                output_pass_k_path = args.output_path / str(total_n)
                output_pass_k_path.mkdir(exist_ok=True)
                event_output_path = output_pass_k_path / event_id
                event_output_path.mkdir()
                write_to_file(event_output_path / "buggy.py", buggy_program)
                write_to_file(event_output_path / "groundtruth_fix.py", fixed_programs[0])
                write_to_file(event_output_path / "fix.py", best_fix)
                if best_fix is not None:
                    write_to_file(event_output_path / "diff_buggy_&_fix.py", get_program_diff(buggy_program, best_fix))
                write_json_file(instance_combined_metrics, event_output_path / "instance_combined_metrics.json")

        # Show progress
        summary_repair_metrics = repair_clerks[-1].get_summary()
        pbar.set_description(
            f"Parseable: {summary_repair_metrics['n_parseable_fixes']}, "
            f"Exact-Match: {summary_repair_metrics['n_exact_match_fixes']} "
        )

    # Write output summary to file
    for repair_clerk, n_outputs in zip(repair_clerks, accumulate(args.n_list)):
        output_all["results_summary"][n_outputs] = repair_clerk.get_summary()
    if args.output_path is not None:
        write_json_file(output_all, args.output_path / "results.json")
        print(f"Write results to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, help="Path to the JSON config file.")
    parser.add_argument(
        "--annotation_data_path",
        type=Path,
        help="Path to the annotation data file in JSON format.",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        nargs="?",
        default="training,calibration,test",
        help="Comma-separated partitions of data to repair (validation or test).",
    )
    parser.add_argument(
        "--instruction_type",
        type=str,
        nargs="?",
        default="general",
        help="Type of instruction for Codex-E (e.g. 'general').",
    )
    parser.add_argument(
        "--n_list",
        type=str,
        nargs="?",
        default="[10]",
        help="List of `n` for generating candidate repairs.",
    )
    parser.add_argument(
        "--temperature_list",
        type=str,
        nargs="?",
        default="[0.5]",
        help="List of `temperature` for generating candidate repairs.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to the output folder.",
    )

    args = parser.parse_args()
    verify_arguments(args)
    main(args)
