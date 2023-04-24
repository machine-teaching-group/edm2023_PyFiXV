import argparse
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.append(".")

from src.utils.IO_utils import load_json_file


def load_grand_technique_results(technique_grand_folder: Path) -> List[Dict[str, Any]]:
    results = []
    technique_folders = sorted([Path(folder) for folder in glob(f"{technique_grand_folder}/*")])
    for folder in technique_folders:
        result_json_path = folder / "results.json"
        result = load_json_file(result_json_path)
        results.append(result)

    return results


def load_validation_folder(validation_folder: Path) -> List[Dict[str, Any]]:
    fold_results = []
    fold_folders = sorted([Path(folder) for folder in glob(f"{validation_folder}/*")])
    for i_fold, fold_folder in enumerate(fold_folders):
        fold_results.append({})
        cal_folder = fold_folder / "cal"
        fold_results[-1]["cal"] = load_grand_technique_results(cal_folder)
        test_folder = fold_folder / "test"
        fold_results[-1]["test"] = load_grand_technique_results(test_folder)

    return fold_results


def extract_param_set_with_max_coverage(results: List[Dict[str, Any]], precision: float, technique: str):
    max_coverage = 0
    real_precision, br_n_list, br_temperature_list, br_threshold = None, None, None, None

    for technique_results in results:
        if technique_results["arguments"][technique]:
            for pass_threshold, perf in technique_results["result_summary"].items():
                if perf["precision"] is not None and perf["precision"] >= precision and perf["coverage"] > max_coverage:
                    max_coverage = perf["coverage"]
                    real_precision = perf["precision"]
                    br_n_list = technique_results["arguments"]["back_repair_n_list"]
                    br_temperature_list = technique_results["arguments"]["back_repair_temperature_list"]
                    br_threshold = pass_threshold

    return {
        "technique": technique,
        "br_n_list": br_n_list,
        "br_temperature_list": br_temperature_list,
        "br_threshold": br_threshold,
        "coverage": round(max_coverage, 2),
        "precision": round(real_precision, 2),
    }


def get_perf(
    results: Sequence[Dict[str, Any]],
    technique: str,
    br_n_list: Sequence[int],
    br_temperature_list: Sequence[float],
    br_threshold: int,
    test_size: int,
) -> Optional[Dict[str, Optional[float]]]:
    for technique_results in results:
        if (
            technique_results["arguments"][technique]
            and technique_results["arguments"]["back_repair_n_list"] == br_n_list
            and technique_results["arguments"]["back_repair_temperature_list"] == br_temperature_list
        ):
            n_present_examples = len(technique_results["results"])
            perf = {
                "coverage": technique_results["result_summary"][str(br_threshold)]["coverage"]
                * n_present_examples
                / test_size,
                "precision": technique_results["result_summary"][str(br_threshold)]["precision"],
            }
            return perf

    return {
        "coverage": None,
        "precision": None,
    }


def compute_test_perf(
    validation_folder: Path, precision_thresholds: Sequence[float], techniques: Sequence[str], test_size: int
):
    perf_by_cal_precision = {}

    results = load_validation_folder(validation_folder=validation_folder)
    for i_fold, fold_result in enumerate(results):
        # print()
        # print(f"=== FOLD {i_fold} ===")
        # print()
        for precision in precision_thresholds:
            # print(f"--- Precision {precision} ---")
            for technique in techniques:
                best_cal_params = extract_param_set_with_max_coverage(
                    results=fold_result["cal"], precision=precision, technique=technique
                )
                # print(f"{technique} | prec {precision:.1f}, {best_cal_params}")
                test_result = get_perf(
                    results=fold_result["test"],
                    technique=technique,
                    br_n_list=best_cal_params["br_n_list"],
                    br_temperature_list=best_cal_params["br_temperature_list"],
                    br_threshold=best_cal_params["br_threshold"],
                    test_size=test_size,
                )
                # print(test_result)
                # print()

                if precision not in perf_by_cal_precision:
                    perf_by_cal_precision[precision] = {}
                if technique not in perf_by_cal_precision[precision]:
                    perf_by_cal_precision[precision][technique] = {}
                if "coverage" not in perf_by_cal_precision[precision][technique]:
                    perf_by_cal_precision[precision][technique]["coverage"] = []
                if "precision" not in perf_by_cal_precision[precision][technique]:
                    perf_by_cal_precision[precision][technique]["precision"] = []

                perf_by_cal_precision[precision][technique]["coverage"].append(test_result["coverage"])
                perf_by_cal_precision[precision][technique]["precision"].append(test_result["precision"])

    # pprint(perf_by_cal_precision)
    return perf_by_cal_precision


def nonemean(lst, weights=None):
    if weights is None:
        weights = [1] * len(lst)

    nonelst, noneweights = zip(*([(v, w) for v, w in zip(lst, weights) if v is not None]))
    if len(noneweights):
        noneweights = np.array(noneweights) / np.sum(noneweights)
    return np.average(nonelst, weights=noneweights)


def nonestd(lst):
    return np.std([x for x in lst if x is not None], ddof=1)


def main(args):
    perf_by_cal_precision = compute_test_perf(
        validation_folder=args.validation_path,
        precision_thresholds=args.precision_thresholds,
        techniques=args.techniques,
        test_size=args.test_size,
    )

    perf_list = []

    for cal_precision in perf_by_cal_precision:
        for technique in perf_by_cal_precision[cal_precision]:
            coverages = perf_by_cal_precision[cal_precision][technique]["coverage"]
            mean_coverage = nonemean(coverages)
            std_coverage = nonestd(coverages)

            precisions = perf_by_cal_precision[cal_precision][technique]["precision"]
            mean_precision = nonemean(precisions)
            std_precision = nonestd(precisions)

            print(
                f"Cal: Precision {cal_precision:.2%}, "
                f"{technique} => Test: "
                f"Precision {mean_precision:.2%} (std {std_precision:.2%}), "
                f"Coverage {mean_coverage:.2%} (std {std_coverage:.2%})"
            )

            perf_list.append(
                (
                    cal_precision * 100,
                    mean_precision * 100,
                    std_precision * 100,
                    mean_coverage * 100,
                    std_coverage * 100,
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_path",
        type=Path,
        nargs="?",
        help="Path to the validation folder",
    )
    parser.add_argument(
        "--precision_thresholds",
        nargs="+",
        type=float,
        default=[0.4, 0.5, 0.6, 0.7],
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        type=str,
        default=["verify_by_back_repair_exact_match"],
    )
    parser.add_argument(
        "--test_size", type=int, help="Size of the test partition in each Cross-validation round (60 for both datasets)"
    )

    args = parser.parse_args()
    print(f"Args: {args}")

    main(args)
