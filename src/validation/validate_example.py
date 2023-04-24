import argparse
import sys
from pathlib import Path

sys.path.append(".")

from src.utils.IO_utils import load_json_file, write_to_file
from src.utils.program_utils import load_program
from src.validation.methods import verify_by_back_repair


def main(args):
    config = load_json_file(args.config_path)
    buggy_program = load_program(args.buggy_program_path)
    repaired_program = load_program(args.fixed_program_path)
    explanation = load_program(args.explanation_path)
    back_repair_params = load_json_file(args.validation_params_path)

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

    write_to_file(file_path=args.output_path, content=validation_result, new_line=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, help="Path to the JSON config file.")
    parser.add_argument("--buggy_program_path", type=Path, help="Path to the buggy program.")
    parser.add_argument("--fixed_program_path", type=Path, help="Path to the fixed program.")
    parser.add_argument("--explanation_path", type=Path, help="Path to the explanation.")
    parser.add_argument("--py_major", type=int, help="Python major version of the program (either 2 or 3).")
    parser.add_argument(
        "--validation_params_path", type=Path, default=None, help="Hyper-parameter values for the validation method."
    )
    parser.add_argument("--output_path", type=Path, default=None, help="path to save the validation result.")

    args = parser.parse_args()

    main(args)
