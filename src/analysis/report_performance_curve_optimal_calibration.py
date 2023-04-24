import argparse
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np

sys.path.append(".")

from src.utils.IO_utils import load_json_file


def get_PyFiXV_curve_points_optimal_calibration(
    n_fold: int,
    validation_path: Path,
    base_precision: float,
    base_coverage: float,
    fold_size: int,
):
    performance_container = defaultdict(lambda: {"prec": [], "cov": []})
    for fold_id in range(n_fold):
        for path in sorted(glob(f"{validation_path}/fold_{fold_id}/test/*/results.json")):
            # Read the validation result
            result = load_json_file(path)
            # Get temperature
            assert len(result["arguments"]["back_repair_temperature_list"]) == 1
            temperature = result["arguments"]["back_repair_temperature_list"][0]
            # Get the maximum threshold
            assert len(result["arguments"]["back_repair_n_list"]) == 1
            max_threshold = result["arguments"]["back_repair_n_list"][0]
            # Get the number of programs that were successfully repaired
            n_repaired = len(result["results"])

            for threshold in range(1, max_threshold + 1):
                precision = result["result_summary"][str(threshold)]["precision"]
                coverage = result["result_summary"][str(threshold)]["coverage"] * n_repaired / fold_size
                performance_container[(temperature, threshold)]["prec"].append(precision)
                performance_container[(temperature, threshold)]["cov"].append(coverage)

    # Compute the avg performance over all cross-validation rounds
    avg_performance_list = [(base_precision, base_coverage, 0.0, 0.0)]
    for (temperature, threshold), perf in performance_container.items():
        mean_prec = np.mean(perf["prec"])
        mean_cov = np.mean(perf["cov"])
        std_prec = np.std(perf["prec"], ddof=1)
        std_cov = np.std(perf["cov"], ddof=1)
        avg_performance_list.append((mean_prec, mean_cov, std_prec, std_cov))
        # print(f"Temp {temperature}, threshold {threshold:2}: PRECISION {mean_prec:.1%}, COVERAGE {mean_cov:.1%}")

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
        get_PyFiXV_curve_points_optimal_calibration(
            n_fold=args.n_fold,
            validation_path=args.validation_path,
            base_precision=args.base_precision,
            base_coverage=args.base_coverage,
            fold_size=args.fold_size,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--validation_path", type=str, help="Path to the validation folder.")
    parser.add_argument("--base_precision", type=float, help="Precision of the feedback without any validation method.")
    parser.add_argument("--base_coverage", type=float, help="Coverage of the feedback without any validation method.")
    parser.add_argument("--n_fold", type=int, help="Number of cross-validation rounds (folds).")
    parser.add_argument("--fold_size", type=int, help="Size of the test data in each cross-validation round.")

    args = parser.parse_args()
    main(args)
