# Generating High-Precision Feedback for Programming Syntax Errors using Large Language Models

This repository contains the implementation for the algorithm PyFiXV, introduced in the [EDM 2023](https://educationaldatamining.org/edm2023/) paper [Generating High-Precision Feedback for Programming Syntax Errors using Large Language Models](https://arxiv.org/abs/2302.04662).

---

### Overview
The repository is structured as follows:
- `src/`: this folder contains the source code of the project.
- `data/`: this folder contains a small sample data of the project. For both TigerJython and Codeforces datasets, we provide a few sample programs created based on real attempts. Furthermore, the list of URLs to the buggy programs in the Codeforces dataset is given. The TigerJython dataset is not available publicly.
- `output/`: this folder is a placeholder for the output of the project. Illustrative examples can be found here.

The implementation requires Python (version >= 3.9) to run. If to be executed on datasets of Python 2 programs, Python 2.7 is also required. Essential packages can be installed by running `pip install -r requirements.txt`.

---

### Applying PyFiXV on an arbitrary buggy program

PyFiXV requires OpenAI's API keys to run. It is assumed these API keys are placed in `~/.password.json` in the following format:
`
{
    "openai-api-completion": "Your-API-key-for-Codex-Complete",
    "openai-api-edit": "Your-API-key-for-Codex-Edit"
}
`

The script below exemplifies how to apply PyFiXV on any arbitrary buggy program by using few-shot examples from the TigerJython dataset.
```
python src/end_to_end/end2end_program.py \
--program_path data/sample_buggy_program.py \
--data_source TJ
```

Similarly, the script below exemplifies how to apply PyFiXV using few-shot examples from the Codeforces dataset.
```
python src/end_to_end/end2end_program.py \
--program_path data/sample_buggy_program.py \
--data_source CF
```

---

### Applying PyFiXV to another programming language

While we experimented with PyFiXV on Python programs, the technique is not limited to Python and can be adapted to other programming languages. This can be done by changing the prompts appropriately (replacing `Python 2` or `Python 3` with the programming language of choice) and setting up the right tokenizer. The Pygments library, as used in this code, has support for tokenizing multiple popular programming languages.