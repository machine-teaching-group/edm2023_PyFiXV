import argparse
import random
import sys
from collections import abc
from pathlib import Path

from tqdm import tqdm

sys.path.append(".")

from src.explanation.evaluation.metrics import ExplanationMetricsClerk
from src.explanation.generation.explain_pipeline import give_explanation
from src.explanation.generation.preprocess import load_annotation_data
from src.explanation.generation.select_fewshot import FS_Criterion
from src.explanation.generation.utils import (
    load_cached_repairs,
    write_explanation_instance_result,
)
from src.repair.generation.preprocess import strip_program
from src.utils.IO_utils import load_json_file, write_json_file
from src.utils.path_utils import serialize_json_value_path

random.seed(42)


def verify_and_parse_arguments(args):
    if not args.config_path.exists():
        raise ValueError(f"config_path `{args.config_path}` does not exist.")

    if not args.annotation_data_path.exists():
        raise ValueError(f"annotation_data_path `{args.annotation_data_path}` does not exist.")

    if args.cached_repair_path is None:
        print("Generating explanation using GROUND-TRUTH repairs.")
    else:
        print("Generating explanation using GENERATED repairs.")

    args.fewshot_priority_list = eval(args.fewshot_priority_list)
    if not isinstance(args.fewshot_priority_list, abc.Sequence):
        raise ValueError("`fewshot_priority_list` must be a sequence of FS_Criterion")
    for each in args.fewshot_priority_list:
        if not isinstance(each, FS_Criterion):
            raise ValueError("`fewshot_priority_list` must be a sequence of FS_Criterion")

    if not args.output_path.parents[0].exists():
        raise ValueError(f"output folder `{args.output_path.parents[0]}` does not exist.")
    if args.output_path.exists() and any(args.output_path.iterdir()):
        raise ValueError(f"output_path {args.output_path} already existed and is non-empty.")
    args.output_path.mkdir(exist_ok=True)

    return args


def main(args):
    config = load_json_file(args.config_path)
    anno_data_train = load_annotation_data(annotation_data_path=args.annotation_data_path, partitions=["training"])
    anno_data_test = load_annotation_data(annotation_data_path=args.annotation_data_path, partitions=[args.partition])
    assert len(anno_data_test) > 0

    # Initialize container for evaluation
    explanation_clerk = ExplanationMetricsClerk()
    output_all = {"arguments": serialize_json_value_path(vars(args)), "results": {}}

    # Repair & explanation for each validation example
    for example in (pbar := tqdm(anno_data_test)):
        # Get required information
        event_id: str = example["bad_program_EventID"]
        buggy_program = strip_program(example["buggy_program"])
        py_major = example["python_major"]

        # Get the fix: either ground-truth fix or a cached fix
        if args.cached_repair_path is not None:
            cached_repair = load_cached_repairs(
                cached_repair_path=args.cached_repair_path, k=args.repair_pass_k, event_id=event_id
            )
            generated_fix = cached_repair["generated_fix"]
            if generated_fix is not None:
                generated_fix = strip_program(generated_fix)
            summary_repair_metrics = cached_repair["results_summary"]
        else:
            generated_fix = strip_program(example["annotation"][0]["fixed_program"])
            summary_repair_metrics = {"Content": "Repair is taken from the ground-truth"}

        # Compute the explanation
        if generated_fix is not None:
            few_shots, prompt, explanation = give_explanation(
                prompt_type=args.prompt_type,
                buggy_program=buggy_program,
                fixed_program=generated_fix,
                py_major=py_major,
                fewshot_pool=anno_data_train,
                fewshot_priority_list=args.fewshot_priority_list,
                nshot=args.nshot,
                config=config,
            )
        else:
            few_shots = []
            prompt = None
            explanation = None

        # - Explanation metrics
        instance_combined_explanation_metrics = explanation_clerk.add(
            final_generated_explanation=explanation,
        )
        summary_explanation_metrics = explanation_clerk.get_summary()

        # Show progress
        pbar.set_description(f"n: {summary_explanation_metrics['n']} ")

        # Aggregate the evaluation metrics for this instance
        output = {
            "explanation": instance_combined_explanation_metrics,
        }
        output_all["results"][event_id] = output

        # Write fixed program, prompt + explanation, etc. to file
        write_explanation_instance_result(
            output_path=args.output_path,
            event_id=event_id,
            buggy_program=buggy_program,
            generated_fix=generated_fix,
            prompt=prompt,
            generated_explanation=explanation,
            few_shots=few_shots,
        )

    # Record aggregated stats
    output_all["results_summary"] = {
        "total_n": pbar.n,
        "repair": summary_repair_metrics,
        "explanation": explanation_clerk.get_summary(),
    }

    # Write output to file
    write_json_file(output_all, args.output_path / "results.json")
    print(f"Write results to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=Path, nargs="?", default="config.json", help="Path to the JSON config file."
    )
    parser.add_argument(
        "--annotation_data_path",
        type=Path,
        nargs="?",
        default="artifacts/annotation_data.json",
        help="Path to the annotation data file in JSON format.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        nargs="?",
        default="calibration",
        help="Partition of data to repair (calibration or test).",
    )
    parser.add_argument(
        "--cached_repair_path",
        type=Path,
        nargs="?",
        default=None,
        help="[Optional] Path to the folder containing cached repairs.",
    )
    parser.add_argument(
        "--repair_pass_k",
        type=int,
        nargs="?",
        default=10,
        help="Only useful when `cached_repair_path` is specified. Select which pass@k to load.",
    )
    parser.add_argument(
        "--nshot", type=int, nargs="?", default=3, help="Number of few-shot examples to add to the prompt."
    )
    parser.add_argument("--prompt_type", type=str, nargs="?", default="buggy_and_diff", help="Type of prompt format.")
    parser.add_argument(
        "--fewshot_priority_list",
        type=str,
        nargs="?",
        default="[FS_Criterion.ERROR_TYPE,FS_Criterion.MASKED_TOKEN_DIFF,FS_Criterion.MASKED_DIFF,FS_Criterion.RANDOM]",
        help="How to select and order the few-shot examples.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to the output folder.",
    )

    args = parser.parse_args()
    args = verify_and_parse_arguments(args)
    main(args)
