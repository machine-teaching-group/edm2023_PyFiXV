from copy import deepcopy
from enum import Enum
from typing import Any, List, Sequence, Tuple, Union

import numpy as np


def compute_edit_distance(s1: Sequence, s2: Sequence) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


class Edit(str, Enum):
    Keep = "Keep"
    Replace = "Replace"
    Remove = "Remove"
    Add = "Add"


def compute_edist_trace(
    s1: List[Any], s2: List[Any], get=lambda x: x
) -> Tuple[int, List[Tuple[Edit, Union[str, Tuple[str, str]]]]]:
    """
    Compute the edit distance between 2 sequences and backtrace on how to change s1 to make it the same as s2.
    """
    assert all(each is not None for each in s1) and all(
        each is not None for each in s2
    ), "All elements need to be not None."

    # prepare 2 arrays
    arr1, arr2 = deepcopy(list(s1)), deepcopy(list(s2))
    arr1.insert(0, None)
    arr2.insert(0, None)

    # initialize the dp table
    distances = np.zeros((len(arr1), len(arr2)), dtype=int)
    for i in range(distances.shape[0]):
        distances[i, 0] = i
    for j in range(distances.shape[1]):
        distances[0, j] = j

    # compute the dp table
    for i, v1 in enumerate(arr1[1:], start=1):
        for j, v2 in enumerate(arr2[1:], start=1):
            if get(v1) == get(v2):
                distances[i, j] = distances[i - 1, j - 1]
            else:
                distances[i, j] = 1 + min(distances[i - 1, j - 1], distances[i - 1, j], distances[i, j - 1])

    edit_distance = distances[len(arr1) - 1, len(arr2) - 1]

    # get the backtrace
    backtrace = []
    i, j = len(arr1) - 1, len(arr2) - 1
    while i + j > 0:
        if i > 0 and j > 0:
            min_around = min(distances[i - 1, j - 1], distances[i - 1, j], distances[i, j - 1])

        if i == 0:
            backtrace.append((Edit.Add, arr2[j]))
            j -= 1
        elif j == 0:
            backtrace.append((Edit.Remove, arr1[i]))
            i -= 1
        elif distances[i - 1, j] == min_around and distances[i - 1, j] + 1 == distances[i, j]:
            backtrace.append((Edit.Remove, arr1[i]))
            i -= 1
        elif distances[i, j - 1] == min_around and distances[i, j - 1] + 1 == distances[i, j]:
            backtrace.append((Edit.Add, arr2[j]))
            j -= 1
        elif distances[i - 1, j - 1] == min_around and distances[i - 1, j - 1] + 1 == distances[i, j]:
            backtrace.append((Edit.Replace, (arr1[i], arr2[j])))
            i -= 1
            j -= 1
        else:
            assert distances[i - 1, j - 1] == distances[i, j] and get(arr1[i]) == get(arr2[j])
            backtrace.append((Edit.Keep, arr2[j]))
            i -= 1
            j -= 1
    forwardtrace = list(reversed(backtrace))

    return edit_distance, forwardtrace
