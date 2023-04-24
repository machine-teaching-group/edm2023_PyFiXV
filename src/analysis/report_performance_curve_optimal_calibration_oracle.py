import argparse
from typing import Sequence

import numpy as np


def cal_opt_coverage(prec, npos, fold_size):
    max_absolute_coverage = min(fold_size, int(npos / prec))
    opt_cov = max_absolute_coverage / fold_size
    real_precision = min(1, npos / max_absolute_coverage)
    return real_precision, opt_cov


def get_optimal_coverage(precision: float, test_positives: Sequence[int], fold_size: int):
    real_precs = []
    opt_covs = []
    for npos in test_positives:
        real_prec, opt_cov = cal_opt_coverage(prec=precision, npos=npos, fold_size=fold_size)
        real_precs.append(real_prec)
        opt_covs.append(opt_cov)

    return np.mean(real_precs), np.mean(opt_covs), np.std(real_precs, ddof=1), np.std(opt_covs, ddof=1)


def get_PyFiXOpt_curve_points_optimal_calibration(
    test_positives: Sequence[int], test_repaired_counts: Sequence[int], fold_size: int
):

    base_precision = np.mean([p / c for p, c in zip(test_positives, test_repaired_counts)])
    base_coverage = np.mean([c / fold_size for c in test_repaired_counts])

    prange = np.arange(round(base_precision, 2), 1.0, 0.01)

    opt_performance = []
    for precision in prange:
        if precision <= base_precision:
            opt_performance.append((base_precision, base_coverage, 0.0, 0.0))
        else:
            real_prec, opt_cov, std_real_prec, std_opt_cov = get_optimal_coverage(
                precision=precision, test_positives=test_positives, fold_size=fold_size
            )
            opt_performance.append((real_prec, opt_cov, std_real_prec, std_opt_cov))

    return opt_performance


def main(args):
    print(
        get_PyFiXOpt_curve_points_optimal_calibration(
            test_positives=args.test_positives,
            test_repaired_counts=args.test_repaired_counts,
            fold_size=args.fold_size,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_positives",
        nargs="+",
        type=int,
        default=[29, 40, 30, 32],
        help="Number of good feedback (without any validation method) per cross-validation round",
    )
    parser.add_argument(
        "--test_repaired_counts",
        nargs="+",
        type=int,
        default=[58, 60, 59, 60],
        help="Number of good repairs per cross-validation round",
    )
    parser.add_argument("--fold_size", type=int, help="Size of the test data in each cross-validation round.")

    args = parser.parse_args()
    main(args)
