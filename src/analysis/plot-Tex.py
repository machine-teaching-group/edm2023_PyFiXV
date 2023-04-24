import argparse
import sys
from pathlib import Path

sys.path.append(".")

import matplotlib.pyplot as plt

from src.analysis.report_performance_curve_optimal_calibration import (
    get_PyFiXV_curve_points_optimal_calibration,
)
from src.analysis.report_performance_curve_optimal_calibration_oracle import (
    get_PyFiXOpt_curve_points_optimal_calibration,
)
from src.analysis.report_performance_curve_optimal_calibration_rule import (
    get_PyFiXRule_curve_points_optimal_calibration,
)

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

# ==============================================================================================


def main(args):
    PyFiXV_data = get_PyFiXV_curve_points_optimal_calibration(
        n_fold=args.n_fold,
        validation_path=args.validation_path,
        base_precision=args.base_precision,
        base_coverage=args.base_coverage,
        fold_size=args.fold_size,
    )
    PyFiXRule_data = get_PyFiXRule_curve_points_optimal_calibration(
        explanation_path=args.explanation_path,
        n_fold=args.n_fold,
        fold_size=args.fold_size,
    )
    PyFiXOpt_data = get_PyFiXOpt_curve_points_optimal_calibration(
        test_positives=args.test_positives,
        test_repaired_counts=args.test_repaired_counts,
        fold_size=args.fold_size,
    )

    if args.data == "TJ":
        data_name = "TigerJython"
        val_yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        val_ylim_top = 1.01
        val_ylim_bottom = 0.10
        val_xticks = [0.4, 0.6, 0.8, 1.0]
        val_xlim_right = 1.05
        val_xlim_left = 0.35
    else:
        data_name = "Codeforces"
        val_yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        val_ylim_top = 1.01
        val_ylim_bottom = 0.10
        val_xticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        val_xlim_right = 1.05
        val_xlim_left = 0.5

    plt.figure(3, figsize=fig_size)

    plt.errorbar(
        [precision for precision, _, _, _ in PyFiXOpt_data],
        [coverage for _, coverage, _, _ in PyFiXOpt_data],
        label=data_name + r": $\textsc{PyFiX-Opt}$",
        color="#FF8849",
        ls="dashed",
        lw=5.5,
    )

    plt.errorbar(
        [precision for precision, _, _, _ in PyFiXV_data],
        [coverage for _, coverage, _, _ in PyFiXV_data],
        label=data_name + r": $\textsc{PyFiXV}$",
        color="g",
        ls="solid",
        lw=3.5,
        marker=".",
        markersize=19,
    )

    plt.errorbar(
        [prec for prec, _, _, _ in PyFiXRule_data],
        [cov for _, cov, _, _ in PyFiXRule_data],
        label=data_name + r": $\textsc{PyFiX-Rule}$",
        color="#0a81ab",
        ls=":",
        lw=5.5,
    )

    plt.xlabel("Precision")
    plt.ylabel("Coverage")
    plt.xticks(val_xticks)
    plt.yticks(val_yticks)
    plt.ylim(top=val_ylim_top, bottom=val_ylim_bottom)
    plt.xlim(right=val_xlim_right, left=val_xlim_left)

    plt.tick_params(axis="both", which="major", pad=10)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.6), ncol=1, fancybox=False, shadow=False)
    plt.savefig(args.output_path / ("plot_" + f"{data_name}.pdf"), format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Either TJ or CF")
    parser.add_argument("--validation_path", type=str, help="Path to the validation folder.")
    parser.add_argument("--explanation_path", type=str, help="Path to the explanation folder.")
    parser.add_argument("--base_precision", type=float, help="Precision of the feedback without any validation method.")
    parser.add_argument("--base_coverage", type=float, help="Coverage of the feedback without any validation method.")
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
