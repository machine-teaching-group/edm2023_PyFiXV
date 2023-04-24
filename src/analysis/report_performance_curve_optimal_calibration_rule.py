import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(".")

from src.validation.evaluation.metrics import calc_metrics
from src.validation.preprocess import load_explanation_df


def nonemean(lst):
    nonelst = [x for x in lst if x is not None]
    if len(nonelst):
        return np.mean(nonelst)
    else:
        return 0


def nonestd(lst, ddof=1):
    nonelst = [x for x in lst if x is not None]
    if len(nonelst):
        return np.std(nonelst, ddof=ddof)
    else:
        return 0


def get_PyFiXRule_curve_points_optimal_calibration(
    explanation_path: Path,
    n_fold: int,
    fold_size: int,
):
    performance_container = defaultdict(lambda: {"prec": [], "cov": []})
    for fold_id in range(n_fold):
        explanation_fold_path = Path(f"{explanation_path}/fold_{fold_id}/test/PyFiX/")
        test_explanation_df = load_explanation_df(
            explanation_grade_path=explanation_fold_path / "human_grade.json",
            path_to_explanation_folder=explanation_fold_path / "explanation_given",
        )

        for x_length in range(30, 1000, 10):
            test_explanation_df["display"] = test_explanation_df["explanation_words"] <= x_length
            coverage, precision, _ = calc_metrics(test_explanation_df, n_total_examples=fold_size)
            performance_container[x_length]["prec"].append(precision)
            performance_container[x_length]["cov"].append(coverage)

    # Compute the avg performance over all cross-validation rounds
    avg_performance_list = []
    for x_length, perf in performance_container.items():
        mean_prec = nonemean(perf["prec"])
        mean_cov = nonemean(perf["cov"])
        std_prec = nonestd(perf["prec"], ddof=1)
        std_cov = nonestd(perf["cov"], ddof=1)
        avg_performance_list.append((mean_prec, mean_cov, std_prec, std_cov))
        # print(f"Explanation length {x_length:2}: PRECISION {mean_prec:.1%}, COVERAGE {mean_cov:.1%}")

    # Compute the points in the curve of optimal calibration
    optimal_calibration_performance_list = []
    last_prec, last_cov = -1, -1
    for perf in sorted(avg_performance_list, reverse=True):
        prec, cov, std_prec, std_cov = perf
        if cov >= last_cov and (abs(prec - last_prec) > 1e-9 or abs(cov - last_cov) > 1e-9):
            optimal_calibration_performance_list.append(perf)
            # print(perf)
            last_prec, last_cov = prec, cov

    return optimal_calibration_performance_list


def main(args):
    print(
        get_PyFiXRule_curve_points_optimal_calibration(
            explanation_path=args.explanation_path,
            n_fold=args.n_fold,
            fold_size=args.fold_size,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--explanation_path", type=str, help="Path to the explanation folder.")
    parser.add_argument("--n_fold", type=int, help="Number of cross-validation rounds (folds).")
    parser.add_argument("--fold_size", type=int, help="Size of the test data in each cross-validation round.")

    args = parser.parse_args()
    main(args)
