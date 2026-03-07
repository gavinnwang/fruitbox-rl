from __future__ import annotations

from typing import List, Tuple
import numpy as np

Action = Tuple[int, int, int, int]


def build_prefix_sum(board: np.ndarray) -> np.ndarray:
    """
    note: prefix sum is 1 indexed
    """
    rows, cols = board.shape
    ps = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    ps[1:, 1:] = board.cumsum(axis=0).cumsum(axis=1)
    return ps


def rect_sum(ps: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
    return ps[r2 + 1, c2 + 1] - ps[r1, c2 + 1] - ps[r2 + 1, c1] + ps[r1, c1]


def find_valid_rectangles(board: np.ndarray, target: int = 10) -> List[Action]:
    rows, cols = board.shape
    actions: List[Action] = []

    for r1 in range(rows):
        colsum = np.zeros(cols, dtype=np.int32)

        for r2 in range(r1, rows):
            colsum += board[r2]

            c1 = 0
            window_sum = 0

            for c2 in range(cols):
                window_sum += int(colsum[c2])

                while c1 <= c2 and window_sum > target:
                    window_sum -= int(colsum[c1])
                    c1 += 1

                if window_sum == target:
                    actions.append((r1, c1, r2, c2))

                    k = c1
                    while k < c2 and colsum[k] == 0:
                        k += 1
                        actions.append((r1, k, r2, c2))

    return actions
