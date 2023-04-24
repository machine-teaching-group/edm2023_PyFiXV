import argparse
import sys
from pathlib import Path

sys.path.append(".")

from src.explanation.generation.explain_pipeline import give_explanation
from src.explanation.generation.preprocess import load_annotation_data
from src.explanation.generation.select_fewshot import FS_Criterion  # noqa: F401
from src.repair.generation.repair_pipeline import repair
from src.utils.IO_utils import load_json_file, write_to_file
from src.utils.program_utils import load_program
from src.validation.methods import verify_by_back_repair


def main(args):
    config = load_json_file(args.config_path)
    buggy_program = load_program(args.program_path)

    # Generate the repaired program
    repaired_program = repair(
        buggy_program=buggy_program,
        py_major=args.py_major,
        config=config,
        instruction_type=args.instruction_type,
    )

    # Write down the repaired program
    repaired_program_path = args.output_folder / "repaired_program.py"
    write_to_file(file_path=repaired_program_path, content=repaired_program, new_line=False)

    # If the repaired program is not None, generate the explanation and do validation
    if repaired_program is not None:
        # Infer the annotation_data_path
        if args.py_major == 2:
            annotation_data_path = Path("data/TigerJython_data.json")
        elif args.py_major == 3:
            annotation_data_path = Path("data/Codeforces_data.json")
        else:
            raise ValueError(f"`py_major` needs to be either 2 or 3, not {args.py_major}")

        # Generate the explanation
        few_shots, prompt, explanation = give_explanation(
            prompt_type=args.prompt_type,
            buggy_program=buggy_program,
            fixed_program=repaired_program,
            py_major=args.py_major,
            fewshot_pool=load_annotation_data(annotation_data_path=annotation_data_path, partitions=("training",)),
            fewshot_priority_list=eval(args.fewshot_priority_list),
            nshot=args.nshot,
            config=config,
        )

        # Write down the explanation
        explanation_path = args.output_folder / "explanation.txt"
        write_to_file(file_path=explanation_path, content=explanation, new_line=False)

        # Do validation
        back_repair_params = {  # a sample hyper-parameter vector for validation
            "n_list": [10],
            "temp_list": [0.5],
            "threshold": 5,
        }
        if verify_by_back_repair(
            config=config,
            buggy_program=buggy_program,
            fixed_program=repaired_program,
            generated_explanation=explanation,
            py_major=args.py_major,
            n_list=back_repair_params["n_list"],
            temperature_list=back_repair_params["temp_list"],
            pass_threshold=back_repair_params["threshold"],
        ).ExactMatch:
            validation_result = "Accepted"
        else:
            validation_result = "Rejected"

        # Write down the validation
        validation_path = args.output_folder / "validation.txt"
        write_to_file(file_path=validation_path, content=validation_result, new_line=False)

    print(f"Please find the output in folder `{args.output_folder}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=Path, help="Path to the buggy program.")
    parser.add_argument("--data_source", type=str, help="Data from few-shot examples. Either 'TJ' or 'CF'.")
    parser.add_argument("--config_path", type=Path, default="src/config.json", help="Path to the JSON config file.")
    parser.add_argument(
        "--instruction_type", type=str, default="general", help="Type of instruction for Codex-E (e.g. 'general')."
    )
    parser.add_argument(
        "--fewshot_priority_list",
        type=str,
        default="[FS_Criterion.ERROR_TYPE,FS_Criterion.MASKED_TOKEN_DIFF,FS_Criterion.MASKED_DIFF,FS_Criterion.RANDOM]",
        help="Type of prompt format.",
    )
    parser.add_argument("--nshot", type=int, default=3, help="Number of few-shot examples to add to the prompt.")
    parser.add_argument("--prompt_type", type=str, default="buggy_and_diff", help="Type of prompt format.")
    parser.add_argument(
        "--output_folder", type=Path, default=Path("output"), help="path to save the generated feedback."
    )

    args = parser.parse_args()

    if args.data_source.upper() not in {"TJ", "CF"}:
        raise ValueError("`data_source` should be either 'TJ' or 'CF'")
    if args.data_source.upper() == "TJ":
        args.py_major = 2
    else:
        args.py_major = 3

    main(args)
