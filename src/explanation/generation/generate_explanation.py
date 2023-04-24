import sys
from typing import Sequence

sys.path.append(".")

from src.utils.codex_utils import get_codex_complete


def generate_explanation(
    prompt: str,
    n_list: Sequence[int] = (1,),
    temperature_list: Sequence[float] = (0,),
):
    assert len(n_list) == len(temperature_list)

    explanation_list = []

    for n, temperature in zip(n_list, temperature_list):
        codex_explanation_results = get_codex_complete(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop="[END]",
            presence_penalty=0,
            frequency_penalty=0,
            max_tokens=256,
            logprobs=None,
        )
        explanation_list += codex_explanation_results

    striped_explanation_list = []
    for explanation in explanation_list:
        if explanation.startswith(" "):
            explanation = explanation[1:]
        if explanation.endswith(" "):
            explanation = explanation[:-1]
        striped_explanation_list.append(explanation)

    return striped_explanation_list


if __name__ == "__main__":
    pass
