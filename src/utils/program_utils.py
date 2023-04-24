import ast
import json
import os
import re
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import astor
import pygments
from pygments.lexers.python import PythonLexer

sys.path.append(".")

from src.utils.edit_distance_utils import Edit, compute_edist_trace  # noqa: E402
from src.utils.IO_utils import write_to_file  # noqa: E402


def load_program(program_path: Union[Path, str]):
    """Load and return a program"""
    with open(program_path, "r") as f:
        program = f.read()
    return program


def parse_python_3(program: str):
    parse_output = {"error_type": "", "error_text": "", "error_line": -1, "error_offset": -1, "source": None}
    try:
        parsed = ast.parse(program)
        parse_output["source"] = astor.to_source(parsed)
    except SyntaxError as e:
        parse_output["error_type"] = e.msg
        parse_output["error_text"] = e.text
        parse_output["error_line"] = e.lineno
        parse_output["error_offset"] = e.offset
        parse_output = deduce_error_text(parse_output=parse_output, program=program)
    return parse_output


def parse_python_2(program: str):
    tmp_file = Path(f"/tmp/tmp_parse_python_2_{time.time()}.py")
    write_to_file(tmp_file, program, new_line=False)
    for _ in range(3):  # try to parse at most 3 times
        try:
            parse_output = json.loads(subprocess.getoutput(f"python2 src/utils/py2_ast_parse.py {tmp_file}"))
            break
        except:  # noqa: E722
            print(f"Cannot AST-parse file {tmp_file}")
            time.sleep(1)

    parse_output = deduce_error_text(parse_output=parse_output, program=program)
    tmp_file.unlink()
    return parse_output


def deduce_error_text(parse_output, program: str):
    if parse_output["error_type"] != "" and parse_output["error_text"] is None:
        if isinstance(parse_output["error_line"], int):
            parse_output["error_text"] = program.split("\n")[parse_output["error_line"] - 1]
        else:
            raise ValueError("buggy program has neither error_text nor error_line")
    return parse_output


def get_parsing_error_string(buggy_program, py_major: int):
    if py_major not in {2, 3}:
        raise ValueError("py_major must be either 2 or 3")
    if py_major == 2:
        parse_output = parse_python_2(program=buggy_program)
    else:
        parse_output = parse_python_3(program=buggy_program)

    error_string = (
        f"Line {parse_output['error_line']}\n{parse_output['error_text']}\nSyntaxError: {parse_output['error_type']}"
    )
    return error_string


def get_parsing_error_message(program_path: str, converse=False, conversion_folder=None):
    """
    Parse the AST of the program, returns the error message if there is, return None otherwise.
    Args:
        - program_path: path to the program.
        - converse: whether to try conversing the program from Python 2 to Python 3.
        - conversion_folder: only effective when converse is True. If conversion_folder is not None,
        it has to be a path to a folder that will contain the conversed Python 3 programs, the path
        to the conversed program will also be returned. If conversion_folder is None, no path will
        be returned.
    """
    program = load_program(program_path)
    error_message = parse_python_3(program)

    if conversion_folder is None:
        py3_folder = "/tmp"
    else:
        py3_folder = conversion_folder

    if error_message["error_type"] != "" and converse is True:
        # converse from py2 to py3 and parse again
        p = subprocess.Popen(
            f"2to3 {program_path} -w -n -o {py3_folder} > /dev/null 2>&1",
            stdout=subprocess.PIPE,
            shell=True,
        )
        p.communicate()
        submission_id = Path(program_path).stem
        py3_program_path = f"{py3_folder}/{submission_id}.py"
        if os.path.exists(py3_program_path):
            py3_program = load_program(py3_program_path)
            py3_error_message = parse_python_3(py3_program)
            if not conversion_folder:
                os.remove(py3_program_path)
        else:
            py3_error_message = error_message

    if error_message["error_type"] == "":
        if converse and conversion_folder:
            return None, program_path
        else:
            return None
    if "py3_error_message" in locals() and py3_error_message["error_type"] == "":
        if converse and conversion_folder:
            return None, py3_program_path
        else:
            return None
    if converse and conversion_folder:
        return error_message, program_path
    else:
        return error_message


lexer = PythonLexer()


def lex_program(program: str):
    """Use pygments.lexers.python.PythonLexer to lex the given program"""
    return list(lexer.get_tokens(program))


def tokenize_program_basic(program: str):
    """
    Lex the program and do some basic filters:
        - replace unnecessary runs of spaces with single spaces.
        - drop all comments.
    """
    filtered_lex_output = []

    if isinstance(program, float) or program is None or len(program) == 0:
        return []

    lex_output = lex_program(program)
    # clean the lexes a little
    is_start_of_line = True
    for component in lex_output:
        token_type, token_value = component
        if token_type == pygments.token.Text and token_value == "\n":
            is_start_of_line = True
            filtered_lex_output.append(component)
        elif not is_start_of_line and token_type == pygments.token.Text and re.match(r"^\s+$", token_value):
            pass  # drop all unnecessary spaces (i.e. all space tokens after the first non-space token in every line)
        elif token_type in pygments.token.Comment.subtypes:
            pass  # drop all comments
        else:
            filtered_lex_output.append(component)
            is_start_of_line = False

    return filtered_lex_output


def program_to_essential_tokens(program: str, strip_chars="\n\r\t\f ") -> List[str]:
    """
    Simplify the program by removing unnecessary tokens, including:
        - comments
        - all spaces after the first non-space token in each line
        - blank lines with/without spaces
        - trailing spaces at the end of the program
    """
    simplified_program_tokens = []

    if isinstance(program, float) or program is None or len(program) == 0:
        return [""]

    lines = program.split("\n")
    meaningful_lines = [line for line in lines if line.strip(strip_chars) != ""]
    program_without_blank_lines = "\n".join(meaningful_lines)

    lex_output = lex_program(program_without_blank_lines)
    is_start_of_line = True
    for component in lex_output:
        token_type, token_value = component
        if token_type == pygments.token.Text and token_value == "\n":
            is_start_of_line = True
            if len(simplified_program_tokens) == 0 or simplified_program_tokens[-1] != "\n":
                simplified_program_tokens.append(token_value)
        elif not is_start_of_line and token_type == pygments.token.Text and re.match(r"^\s+$", token_value):
            pass  # drop all unnecessary spaces (i.e. all space tokens after the first non-space token in every line)
        elif token_type in pygments.token.Comment.subtypes:
            pass  # drop all comments
        else:
            simplified_program_tokens.append(token_value)
            is_start_of_line = False

    while len(simplified_program_tokens) > 0 and simplified_program_tokens[-1].strip(strip_chars) == "":
        del simplified_program_tokens[-1]

    return simplified_program_tokens


def get_program_token_diff(buggy_program: str, fixed_program: str) -> List[Tuple[Edit, Union[str, Tuple[str, str]]]]:
    _, forward_trace = compute_edist_trace(
        program_to_essential_tokens(buggy_program), program_to_essential_tokens(fixed_program)
    )
    token_diff = [element for element in forward_trace if element[0] != Edit.Keep]
    return token_diff


def get_program_masked_token_diff(token_diff: Sequence[Tuple[Edit, Union[str, Tuple[str, str]]]]):
    masked_token_diff = []
    for element in token_diff:
        if isinstance(element[1], str):
            masked_token_diff.append((element[0], mask_reliables(element[1])))
        else:
            assert isinstance(element[1], tuple)  # element[1] is a tuple of 2 strings
            masked_token_diff.append(
                (
                    element[0],
                    (mask_reliables(element[1][0]), mask_reliables(element[1][1])),
                )
            )
    return masked_token_diff


python_keywords = [
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "exec",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "not",
    "or",
    "pass",
    "print",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
]


def mask_reliables(s: str, unreliable_terminals=string.punctuation):
    if all(value in unreliable_terminals for value in s):
        masked = s  # keep unreliable tokens
    elif re.match(r"^\s+$", s):
        masked = s  # keep spaces, tabs, new-lines
    elif s in python_keywords:
        masked = s  # keep keywords
    else:
        masked = "{[<MASKED>]}"  # all other tokens are masked

    return masked


def tokenize_and_mask_reliables(program: str, unreliable_terminals=string.punctuation):
    """
    First, call to `tokenize_program_basic`.
    Then, mask all non-punctuation tokens to the same token.
    This function is different from `tokenize_program_unreliable_terminals` by keeping (not masking) Python keywords.
    """
    filtered_lex_output = tokenize_program_basic(program)
    masked_lex_output = []
    for component in filtered_lex_output:
        token_type, token_value = component
        if all(value in unreliable_terminals for value in token_value):
            masked_lex_output.append(component)  # keep unreliable tokens
        elif token_type == pygments.token.Text and re.match(r"^\s+$", token_value):
            masked_lex_output.append(component)  # keep spaces, tabs, new-lines
        elif token_value in python_keywords:
            masked_lex_output.append(component)  # keep keywords
        else:
            masked_component = (None, "<MASKED>")  # all other tokens are masked
            masked_lex_output.append(masked_component)

    return masked_lex_output


def get_program_diff_by_path(program_path_1: Path, program_path_2: Path):
    bash_output = subprocess.run(["diff", program_path_1, program_path_2], capture_output=True)
    if bash_output.returncode != 1:
        raise Exception(
            f"During executing `get_program_diff('{program_path_1}', '{program_path_2}') get an error: "
            + str(bash_output)
        )
    return bash_output.stdout.decode("utf8")


def get_program_diff(program_1: str, program_2: str):
    tmp_path_1 = Path(f"/tmp/tmp_1_for_program_diff_{time.time()}.py")
    tmp_path_2 = Path(f"/tmp/tmp_2_for_program_diff_{time.time()}.py")
    write_to_file(tmp_path_1, program_1)
    write_to_file(tmp_path_2, program_2)
    diff = get_program_diff_by_path(tmp_path_1, tmp_path_2)
    tmp_path_1.unlink()
    tmp_path_2.unlink()
    return diff


def get_program_inline_diff(program_1: str, program_2: str):
    tok_program_1 = program_to_essential_tokens(program_1)
    tok_program_2 = program_to_essential_tokens(program_2)
    _, forward_trace = compute_edist_trace(tok_program_1, tok_program_2)

    s = ""
    remove_list, add_list = [], []

    def flush_lists(curr_s):
        nonlocal remove_list, add_list
        if len(remove_list) > 0:
            curr_s += "[Remove]" + "".join(remove_list) + "[End]"
            remove_list = []
        if len(add_list) > 0:
            curr_s += "[Add]" + "".join(add_list) + "[End]"
            add_list = []
        return curr_s

    for edit, value in forward_trace:
        if edit == Edit.Keep:
            s = flush_lists(s)
            s += value
        elif edit == Edit.Remove:
            remove_list.append(value)
        elif edit == Edit.Add:
            add_list.append(value)
        else:  # edit == Edit.Replace
            remove_list.append(value[0])
            add_list.append(value[1])

    s = flush_lists(s)
    assert len(remove_list) + len(add_list) == 0

    return s


if __name__ == "__main__":
    pass
