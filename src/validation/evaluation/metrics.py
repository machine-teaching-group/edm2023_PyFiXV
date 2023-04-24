from collections import namedtuple

import pandas as pd

MetricScore = namedtuple("Score", "coverage precision opt_precision")


def calc_metrics(df: pd.DataFrame, n_total_examples: int) -> MetricScore:
    """
    Return Precision and Coverage of a verfication output.
    The input dataframe `df` needs to have 2 columns:
        `is_correct`: whether the feedback is correct and good to show to the student.
        `display`: whether the verification method decide to show to the student
    """
    assert "display" in df.columns
    n_display = len(df[df["display"] == 1])
    n_correct = len(df[df["is_correct"] == 1])
    n_correct_display = sum(df[df["is_correct"] == 1]["display"])

    # Precision
    if n_display:
        precision = n_correct_display / n_display
    else:
        precision = None
    # Coverage
    coverage = n_display / n_total_examples
    # Optimal Precision (for the current Coverage)
    if n_display:
        opt_precision = min(1.0, n_correct / n_display)
    else:
        opt_precision = None

    return MetricScore(coverage, precision, opt_precision)
