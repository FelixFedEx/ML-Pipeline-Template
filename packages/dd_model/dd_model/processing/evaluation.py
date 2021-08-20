from __future__ import absolute_import

import typing as t


def caculate_prediction_accuracy(y_true, y_pred) -> t.Tuple[float, t.List[int]]:
    possible_dup_obs = [
        [zdx[0] for zdx in ydx] if len(ydx) != 0 else [] for ydx in y_pred
    ]
    correct_prediction = []
    for idx in range(len(y_true)):
        if y_true[idx] in possible_dup_obs[idx]:
            correct_prediction.append(True)
        else:
            correct_prediction.append(False)
    acc = round(sum(correct_prediction) / len(y_true), 3)

    return acc, correct_prediction
