import argparse
import random
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(".")

from src.utils.IO_utils import load_json_file, write_json_file

random.seed(2022)


def validate_args(args) -> None:
    anno_data = load_json_file(args.annotation_data_path)
    if len(anno_data) % args.k_folds != 0:
        raise ValueError(f"The data size ({len(args.anno_data)}) should divide k ({args.k_folds}).")


def normalize_anno(annotation_list):
    # make each instance have only 1 annotation
    for instance in annotation_list:
        instance["annotation"] = instance["annotation"][:1]

    return annotation_list


def validate_data(fold_list, k_folds: int):
    assert len(fold_list) == k_folds

    for fold in fold_list:
        # no data-point is duplicated in a fold
        event_ids = set(instance["bad_program_EventID"] for instance in fold)
        assert len(event_ids) == len(fold)

        # no program has more than 1 annotation
        assert all(len(instance["annotation"]) == 1 for instance in fold)

        # size of the test and calibration partitions equal to size of the whole data divided by k_folds
        fold_size = len(fold) // k_folds
        assert len([instance for instance in fold if instance["partition"] == "calibration"]) == fold_size
        assert len([instance for instance in fold if instance["partition"] == "test"]) == fold_size

    # all folds have the same size
    assert all(len(fold_list[0]) == len(fold_list[i]) for i in range(1, len(fold_list)))

    # the test data from all folds form the whole dataset
    test_ids = set(
        instance["bad_program_EventID"] for fold in fold_list for instance in fold if instance["partition"] == "test"
    )
    assert len(fold_list[0]) == len(
        test_ids
    ), f"Data size ({len(fold_list[0])}) is different from total size of test ({len(test_ids)})"


def main(args):
    anno_data = load_json_file(args.annotation_data_path)
    anno_data = normalize_anno(anno_data)
    random.shuffle(anno_data)
    print(f"Data size: {len(anno_data)}")

    folds = []
    fold_size = len(anno_data) // args.k_folds
    for fold_id in range(args.k_folds):
        test_data = anno_data[fold_id * fold_size : (fold_id + 1) * fold_size]
        train_data = anno_data[: fold_id * fold_size] + anno_data[(fold_id + 1) * fold_size :]
        random.shuffle(train_data)
        fewshot_data = train_data[:-fold_size]
        cal_data = train_data[-fold_size:]
        #
        for instance in fewshot_data:
            instance["partition"] = "training"
        for instance in cal_data:
            instance["partition"] = "calibration"
        for instance in test_data:
            instance["partition"] = "test"

        fold_data = deepcopy(fewshot_data + cal_data + test_data)
        folds.append(fold_data)

        print(
            f"Fold {fold_id}: "
            f"Train size ({len(fewshot_data)}), "
            f"Cal size ({len(cal_data)}), "
            f"Test size ({len(test_data)})"
        )

    validate_data(folds, k_folds=args.k_folds)
    for fold_id, fold_data in enumerate(folds):
        output_file_path = args.output_path / f"anno_fold_{fold_id}.json"
        if output_file_path.exists():
            raise ValueError(f"Output file path {output_file_path} already existed.")
        write_json_file(fold_data, args.output_path / f"anno_fold_{fold_id}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_data_path",
        type=Path,
        help="Path to the annotation data file in JSON format.",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the output folder.",
    )

    args = parser.parse_args()

    validate_args(args)

    main(args)
