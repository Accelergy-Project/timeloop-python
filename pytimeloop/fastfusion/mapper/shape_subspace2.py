"""
General idea:
- Often, higher hardware utilization leads to better metrics
- When that is not the case, the cause is that the hardware resource is
  shared and higher utilization by one user trades off utilization by
  another.

In terms of tile shape, the utilization of interest is:
- Higher buffer utilization due to smaller factor, larger tile shape of
  temporal loops
Spatial loops are more complicated because larger tile shape means lower
parallel hardware utilization.

To keep the shape iterator more generic, we allow tagging of a particular
loop with hints:
- Maximize tile shape
- Minimize tile shape
- Explore all
The maximize/minimize tile shape tags will cause the iterator to attempt
to quickly find the largest/smallest *valid* tile shape.
"""
from collections.abc import Callable
from enum import Enum

from .shape_subspace import ShapeSubspace


class IteratorHint(Enum):
    MAXIMIZE = 0
    MINIMIZE = 1
    EXPLORE  = 2


class FastShapeSubspaceIterator:
    def __init__(self,
                 shape_subspace: ShapeSubspace,
                 hints: list[IteratorHint]):
        self.shape_subspace = shape_subspace
        self.hints = hints

    def explore_idx(self, idx: int):
        hint = self.hints[idx]
        if hint == IteratorHint.MAXIMIZE:
            binary_search(min_val,
                          max_val,
                          evaluator,
                          search_max=True)
        elif hint == IteratorHint.MINIMIZE:
            binary_search(min_val,
                          max_val,
                          evaluator,
                          search_max=False)
        elif hint == IteratorHint.EXPLORE:
            exhaustive(min_val,
                       max_val,
                       evaluator)
            pass
        else:
            raise ValueError(f"Unknown hint {hint}")


class EvaluationResult(Enum):
    TOO_SMALL = 0
    VALID     = 1
    TOO_LARGE = 2


def binary_search(min: int,
                  max: int,
                  evaluate: Callable[[int], EvaluationResult],
                  search_max: bool):
    if min > max:
        raise ValueError("min must be lower or equal to max")

    while min < max - 1:
        cur = (min + max) // 2
        cur_result = evaluate(cur)
        if cur_result == EvaluationResult.TOO_LARGE:
            max = cur - 1
        elif cur_result == EvaluationResult.TOO_SMALL:
            min = cur + 1
        else:
            if search_max:
                min = cur
            else:
                max = cur

    assert min >= max - 1 and min < max
    if search_max:
        evaluate_order = [max, min]
    else:
        evaluate_order = [min, max]
    for cur in evaluate_order:
        if evaluate(cur) == EvaluationResult.VALID:
            return cur
    return None
