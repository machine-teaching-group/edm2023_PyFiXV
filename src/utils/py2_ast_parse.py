#! /usr/bin/python2
import argparse
import ast
import json
from os.path import exists

import astor


def main(args):
    # output structure
    output = {"error_type": "", "error_text": "", "error_line": -1, "error_offset": -1, "source": None}

    # load program
    if not exists(args.program_path):
        # if the path is not existed, return an error
        output["error_type"] = "FileNotExists"
        print(json.dumps(output))
        return
    with open(args.program_path, "r") as f:
        program = f.read()

    # try parsing
    try:
        parsed = ast.parse(program)
        output["source"] = astor.to_source(parsed)
        print(json.dumps(output))
    except SyntaxError as e:
        output["error_type"] = e.msg
        output["error_text"] = e.text
        output["error_line"] = e.lineno
        output["error_offset"] = e.offset

        print(json.dumps(output))
        return

    except Exception as e:
        print(e)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("program_path", type=str, help="Path to the program to parse.")

    args = parser.parse_args()

    main(args)
