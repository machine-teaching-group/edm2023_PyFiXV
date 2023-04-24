import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.append(".")

from src.analysis.report_CV_performance import nonemean, nonestd
from src.validation.evaluation.metrics import calc_metrics
from src.validation.preprocess import load_explanation_df


def compute_test_perf(explanation_folder: Path, precision_thresholds: Sequence[float], test_size: int, n_fold: int):
    for precision_threshold in precision_thresholds:
        print("===========================================")
        print(f"Precision threshold: {precision_threshold}")
        test_precisions, test_coverages = [], []
        for fold_id in range(n_fold):
            cal_technique_path = explanation_folder / f"fold_{fold_id}" / "cal" / "PyFiX"
            test_technique_path = explanation_folder / f"fold_{fold_id}" / "test" / "PyFiX"

            cal_explanation_df = load_explanation_df(
                explanation_grade_path=cal_technique_path / "human_grade.json",
                path_to_explanation_folder=cal_technique_path / "explanation_given",
            )
            test_explanation_df = load_explanation_df(
                explanation_grade_path=test_technique_path / "human_grade.json",
                path_to_explanation_folder=test_technique_path / "explanation_given",
            )

            # Best config so far
            best_x_length = None
            # taken_precision = None
            best_coverage = -1

            highest_precision = -1
            highest_p_x_length = None
            # highest_p_coverage = None

            for x_length in range(30, 1000, 10):
                cal_explanation_df["display"] = cal_explanation_df["explanation_words"] <= x_length
                coverage, precision, _ = calc_metrics(cal_explanation_df, n_total_examples=test_size)
                if precision is not None:
                    if precision >= precision_threshold:
                        if coverage > best_coverage:
                            best_x_length = x_length
                            # taken_precision = precision
                            best_coverage = coverage
                    else:
                        if precision > highest_precision:
                            highest_precision = precision
                            highest_p_x_length = x_length
                            # highest_p_coverage = coverage

            if best_x_length is None:  # no x_length found for this precision threshold
                print("Couldn't found rule for this high precision, take the one with the best precision")
                best_x_length = highest_p_x_length
                # taken_precision = highest_precision
                # best_coverage = highest_p_coverage

            print(f"Round {fold_id} uses explanation-length: {best_x_length}")

            # Apply to test data
            test_explanation_df["display"] = test_explanation_df["explanation_words"] <= best_x_length
            test_cov, test_prec, _ = calc_metrics(test_explanation_df, n_total_examples=test_size)
            test_precisions.append(test_prec)
            test_coverages.append(test_cov)

        print(
            f"Test precision: {nonemean(test_precisions):.1%} +- {nonestderr(test_precisions, 4):.1%} "
            f"|| {test_precisions}"
        )
        print(
            f"Test coverage: {nonemean(test_coverages):.1%} +- {nonestderr(test_coverages, 4):.1%} ||  "
            f"{test_coverages}"
        )


def nonestderr(lst, n_fold: int):
    return nonestd(lst) / (n_fold**0.5)


def main(args):
    compute_test_perf(
        explanation_folder=args.explanation_path,
        precision_thresholds=args.precision_thresholds,
        test_size=args.test_size,
        n_fold=args.n_fold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--explanation_path",
        type=Path,
        nargs="?",
        help="Path to the explanation folder",
    )
    parser.add_argument(
        "--precision_thresholds",
        nargs="+",
        type=float,
        default=[0.4, 0.5, 0.6, 0.7],
    )
    parser.add_argument(
        "--test_size", type=int, help="Size of the test partition in each Cross-validation round (60 for both datasets)"
    )
    parser.add_argument("--n_fold", type=int, default=4)

    args = parser.parse_args()
    print(f"Args: {args}")

    main(args)
