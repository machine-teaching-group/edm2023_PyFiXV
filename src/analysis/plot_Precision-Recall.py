import argparse
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.append(".")

import matplotlib.pyplot as plt

from src.utils.IO_utils import load_json_file
from src.validation.evaluation.metrics import calc_metrics
from src.validation.preprocess import load_explanation_df


plt.switch_backend("agg")
import matplotlib as mpl

mpl.rcParams["font.serif"] = ["times new roman"]
mpl.rcParams["text.latex.preamble"] = [r"\usepackage{newtxmath}"]

color_codes = ["r", "g", "b", "#F08080", "#8B0000", "#E802FB", "#C64C23", "#223206", "#7E3391", "#040004"]
color_for_orig = "#8b0000"
color_for_pot = "#F08080"

mpl.rc("font", **{"family": "serif", "serif": ["Times"], "size": 29})
mpl.rc("legend", **{"fontsize": 26.5})
mpl.rc("text", usetex=True)
fig_size = [7, 4.8]

# === PyFiXV curve ===
def get_PyFiXV_curve_points_optimal_calibration(
    n_fold: int,
    validation_path: Path,
    base_precision: float,
    fold_size: int,
    n_positives: Sequence[int],
):
    performance_container = defaultdict(lambda: {"prec": [], "recall": []})
    for fold_id in range(n_fold):
        n_pos = n_positives[fold_id]
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
                recall = precision * coverage * fold_size / n_pos
                performance_container[(temperature, threshold)]["prec"].append(precision)
                performance_container[(temperature, threshold)]["recall"].append(recall)

    # Compute the avg performance over all cross-validation rounds
    avg_performance_list = [(base_precision, 1, 0.0, 0.0)]
    for (temperature, threshold), perf in performance_container.items():
        mean_prec = np.mean(perf["prec"])
        mean_recall = np.mean(perf["recall"])
        std_prec = np.std(perf["prec"], ddof=1)
        std_recall = np.std(perf["recall"], ddof=1)
        avg_performance_list.append((mean_prec, mean_recall, std_prec, std_recall))
        # print(f"Temp {temperature}, threshold {threshold:2}: PRECISION {mean_prec:.1%}, COVERAGE {mean_cov:.1%}")

    # Compute the points in the curve of optimal calibration
    optimal_calibration_performance_list = []
    last_prec, last_recall = -1, -1
    for perf in sorted(avg_performance_list, reverse=True):
        prec, recall, std_prec, std_recall = perf
        if recall >= last_recall and (abs(prec - last_prec) > 1e-9 or abs(recall - last_recall) > 1e-9):
            optimal_calibration_performance_list.append(perf)
            # print(perf)
            last_prec, last_recall = prec, recall

    return optimal_calibration_performance_list


# === PyFiXRule curve ===
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
    n_positives: Sequence[int],
):
    performance_container = defaultdict(lambda: {"prec": [], "recall": []})
    for fold_id in range(n_fold):
        n_pos = n_positives[fold_id]
        explanation_fold_path = Path(f"{explanation_path}/fold_{fold_id}/test/PyFiX/")
        test_explanation_df = load_explanation_df(
            explanation_grade_path=explanation_fold_path / "human_grade.json",
            path_to_explanation_folder=explanation_fold_path / "explanation_given",
        )

        for x_length in range(30, 1000, 10):
            test_explanation_df["display"] = test_explanation_df["explanation_words"] <= x_length
            coverage, precision, _ = calc_metrics(test_explanation_df, n_total_examples=fold_size)
            recall = precision * coverage * fold_size / n_pos
            performance_container[x_length]["prec"].append(precision)
            performance_container[x_length]["recall"].append(recall)

    # Compute the avg performance over all cross-validation rounds
    avg_performance_list = []
    for x_length, perf in performance_container.items():
        mean_prec = nonemean(perf["prec"])
        mean_recall = nonemean(perf["recall"])
        std_prec = nonestd(perf["prec"], ddof=1)
        std_recall = nonestd(perf["recall"], ddof=1)
        avg_performance_list.append((mean_prec, mean_recall, std_prec, std_recall))
        # print(f"Explanation length {x_length:2}: PRECISION {mean_prec:.1%}, COVERAGE {mean_cov:.1%}")

    # Compute the points in the curve of optimal calibration
    optimal_calibration_performance_list = []
    last_prec, last_recall = -1, -1
    for perf in sorted(avg_performance_list, reverse=True):
        prec, recall, std_prec, std_recall = perf
        if recall >= last_recall and (abs(prec - last_prec) > 1e-9 or abs(recall - last_recall) > 1e-9):
            optimal_calibration_performance_list.append(perf)
            # print(perf)
            last_prec, last_recall = prec, recall

    return optimal_calibration_performance_list


# === PyFiXOpt curve ===
def cal_opt_coverage(prec, npos, fold_size):
    max_absolute_coverage = min(fold_size, int(npos / prec))
    opt_cov = max_absolute_coverage / fold_size
    real_precision = min(1, npos / max_absolute_coverage)
    return real_precision, opt_cov


def get_optimal_recall(precision: float, test_positives: Sequence[int], fold_size: int):
    real_precs = []
    opt_recalls = []
    for npos in test_positives:
        real_prec, opt_cov = cal_opt_coverage(prec=precision, npos=npos, fold_size=fold_size)
        opt_recall = min(1, opt_cov * fold_size / npos)
        real_precs.append(real_prec)
        opt_recalls.append(opt_recall)

    return np.mean(real_precs), np.mean(opt_recalls), np.std(real_precs, ddof=1), np.std(opt_recalls, ddof=1)


def get_PyFiXOpt_curve_points_optimal_calibration(
    test_positives: Sequence[int], test_repaired_counts: Sequence[int], fold_size: int
):

    base_precision = np.mean([p / c for p, c in zip(test_positives, test_repaired_counts)])
    base_recall = 1

    prange = np.arange(round(base_precision, 2), 1.0, 0.01)

    opt_performance = []
    for precision in prange:
        if precision <= base_precision:
            opt_performance.append((base_precision, base_recall, 0.0, 0.0))
        else:
            real_prec, opt_recall, std_real_prec, std_opt_recall = get_optimal_recall(
                precision=precision, test_positives=test_positives, fold_size=fold_size
            )
            opt_performance.append((real_prec, opt_recall, std_real_prec, std_opt_recall))

    return opt_performance


# ==============================================================================================


def main(args):
    PyFiXV_data = get_PyFiXV_curve_points_optimal_calibration(
        n_fold=args.n_fold,
        validation_path=args.validation_path,
        base_precision=args.base_precision,
        fold_size=args.fold_size,
        n_positives=args.test_positives,
    )
    PyFiXRule_data = get_PyFiXRule_curve_points_optimal_calibration(
        explanation_path=args.explanation_path,
        n_fold=args.n_fold,
        fold_size=args.fold_size,
        n_positives=args.test_positives,
    )
    PyFiXOpt_data = get_PyFiXOpt_curve_points_optimal_calibration(
        test_positives=args.test_positives,
        test_repaired_counts=args.test_repaired_counts,
        fold_size=args.fold_size,
    )

    if args.data == "TJ":
        data_name = "TigerJython"
        val_xticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        val_xlim_top = 1.01
        val_xlim_bottom = 0.10
        val_yticks = [0.4, 0.6, 0.8, 1.0]
        val_ylim_right = 1.05
        val_ylim_left = 0.35
    else:
        data_name = "Codeforces"
        val_xticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        val_xlim_top = 1.01
        val_xlim_bottom = 0.10
        val_yticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        val_ylim_right = 1.05
        val_ylim_left = 0.5

    plt.figure(3, figsize=fig_size)

    plt.errorbar(
        [recall for _, recall, _, _ in PyFiXOpt_data],
        [precision for precision, _, _, _ in PyFiXOpt_data],
        label=data_name + r": $\textsc{PyFiX-Opt}$",
        color="#FF8849",
        ls="dashed",
        lw=5.5,
    )

    plt.errorbar(
        [recall for _, recall, _, _ in PyFiXV_data],
        [precision for precision, _, _, _ in PyFiXV_data],
        label=data_name + r": $\textsc{PyFiXV}$",
        color="g",
        ls="solid",
        lw=3.5,
        marker=".",
        markersize=19,
    )

    plt.errorbar(
        [recall for _, recall, _, _ in PyFiXRule_data],
        [prec for prec, _, _, _ in PyFiXRule_data],
        label=data_name + r": $\textsc{PyFiX-Rule}$",
        color="#0a81ab",
        ls=":",
        lw=5.5,
    )

    plt.ylabel("Precision")
    plt.xlabel("Recall")
    # plt.xticks(val_xticks)
    # plt.yticks(val_yticks)
    # plt.xlim(right=val_xlim_top, left=val_xlim_bottom)
    # plt.ylim(top=val_ylim_right, bottom=val_ylim_left)
    plt.xlim(right=1, left=0)
    plt.ylim(top=1, bottom=0)
    plt.tick_params(axis="both", which="major", pad=10)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.6), ncol=1, fancybox=False, shadow=False)
    plt.savefig(args.output_path / ("plot_" + f"{data_name}.pdf"), format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Either TJ or CF")
    parser.add_argument("--validation_path", type=str, help="Path to the validation folder.")
    parser.add_argument("--explanation_path", type=str, help="Path to the explanation folder.")
    parser.add_argument("--base_precision", type=float, help="Precision of the feedback without any validation method.")
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
    parser.add_argument("--n_fold", type=int, help="Number of cross-validation rounds (folds).")
    parser.add_argument("--fold_size", type=int, help="Size of the test data in each cross-validation round.")
    parser.add_argument("--output_path", type=Path, help="Path to the output folder")

    args = parser.parse_args()
    assert args.data in {"TJ", "CF"}

    main(args)
