import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.append(".")

from src.repair.generation.generate_fixes import generate_fixes
from src.repair.generation.make_instruction import make_repair_instruction
from src.repair.generation.select_best_fix import select_fix_by_ed
from src.utils.IO_utils import load_json_file, write_to_file
from src.utils.program_utils import load_program


def repair(
    buggy_program: str,
    py_major: int,
    config=None,
    instruction_type: Optional[str] = None,
    instruction: Optional[str] = None,
    n_list: Sequence[int] = (10,),
    temperature_list: Sequence[float] = (0.5,),
):
    # pre-check some arguments
    if buggy_program.strip() == "":
        raise ValueError("Buggy program cannot be empty.")
    if py_major not in {2, 3}:
        raise ValueError("Python major version needs to be either 2 or 3.")
    if (instruction_type is not None) + (instruction is not None) != 1:
        raise ValueError("Please input either instruction_type or instruction, but not both.")
    if instruction_type is not None:
        if instruction_type not in {"general", "specific"}:
            raise ValueError("Supported instruction types are ('general', 'specific').")
        if config is None:
            raise ValueError("Please specify config.")
    if len(n_list) != len(temperature_list):
        raise ValueError("Size of n_list and temperature_list must be the same.")

    # generate the repair instruction
    if instruction_type is not None:
        instruction = make_repair_instruction(instruction_type=instruction_type, py_major=py_major, config=config)

    # generate multiple fixes
    generated_fixes = generate_fixes(
        buggy_program, instruction_list=[instruction] * len(n_list), n_list=n_list, temperature_list=temperature_list
    )

    # select the best fix
    fixed_program, ed, _ = select_fix_by_ed(
        buggy_program=buggy_program, generated_fixes=generated_fixes, py_major=py_major
    )

    return fixed_program


def main(args):
    n_list = eval(args.n_list)
    temperature_list = eval(args.temperature_list)

    repaired_program = repair(
        config=load_json_file(args.config_path),
        buggy_program=load_program(args.program_path),
        py_major=args.py_major,
        instruction_type=args.instruction_type,
        n_list=n_list,
        temperature_list=temperature_list,
    )

    if args.output_path is not None:
        write_to_file(file_path=args.output_path, content=repaired_program, new_line=False)
    else:
        print(repaired_program)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, help="Path to the JSON config file.")
    parser.add_argument("--program_path", type=Path, help="Path to the buggy program.")
    parser.add_argument("--py_major", type=int, help="Python major version of the program (either 2 or 3).")
    parser.add_argument("--instruction_type", type=str, help="Type of instruction for Codex-E (e.g. 'general').")
    parser.add_argument("--n_list", type=str, default="[10]", help="Number of outputs to request from Codex.")
    parser.add_argument(
        "--temperature_list", type=str, default="[0.5]", help="Temperature to request outputs from Codex."
    )
    parser.add_argument("--output_path", type=Path, default=None, help="path to save the generated repaired program.")

    args = parser.parse_args()

    main(args)
