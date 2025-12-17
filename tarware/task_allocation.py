from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Assignment:
    robot: object
    item: object


def _path_distance(env, robot, item) -> int:
    """
    Distance proxy used for allocation: A* path length.
    Returns a large number if no path is found.
    """
    path = env.find_path((robot.y, robot.x), (item.y, item.x), robot, care_for_agents=False)
    if path is None:
        return 10**9
    if len(path) == 0 and (robot.x, robot.y) != (item.x, item.y):
        return 10**9
    return len(path)


def _linear_sum_assignment_min(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Min-cost assignment for a rectangular cost matrix using a pure-Python Hungarian variant.

    Returns (row_ind, col_ind) such that each row is assigned to one col with min total cost.
    This implementation assumes number of rows <= number of cols.
    """
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D array")

    n, m = cost.shape
    if n == 0 or m == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if n > m:
        raise ValueError("Hungarian solver expects rows <= cols")

    # 1-indexed arrays, following the classic O(n^3) implementation.
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)  # which row is matched to column j
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Augmenting
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Build assignment: p[j] = i means row i assigned to col j.
    row_ind = np.arange(n, dtype=int)
    col_ind = np.empty(n, dtype=int)
    for j in range(1, m + 1):
        i = p[j]
        if 1 <= i <= n:
            col_ind[i - 1] = j - 1
    return row_ind, col_ind


def allocate_greedy_fifo(env, available_robots: Sequence, request_queue: Sequence, already_assigned_item_ids: Iterable[int]):
    """
    Greedy FIFO: for each item in queue order, assign the closest available robot.
    """
    assigned_item_ids = set(already_assigned_item_ids)
    remaining = list(available_robots)
    assignments: List[Assignment] = []

    for item in request_queue:
        if item.id in assigned_item_ids:
            continue
        if not remaining:
            break
        dists = [_path_distance(env, r, item) for r in remaining]
        robot = remaining[int(np.argmin(dists))]
        assignments.append(Assignment(robot=robot, item=item))
        assigned_item_ids.add(item.id)
        remaining.remove(robot)
    return assignments


def allocate_batch_min_cost(
    env,
    available_robots: Sequence,
    request_queue: Sequence,
    already_assigned_item_ids: Iterable[int],
    batch_size: int | None = None,
):
    """
    Batch optimal (non-greedy): take the first `batch_size` unassigned items in FIFO order and
    compute a min-total-distance matching between available robots and those items.
    """
    assigned_item_ids = set(already_assigned_item_ids)
    unassigned_items = [item for item in request_queue if item.id not in assigned_item_ids]
    if not unassigned_items or not available_robots:
        return []

    if batch_size is None:
        batch_size = min(len(unassigned_items), len(available_robots))
    batch_size = max(1, int(batch_size))

    items = unassigned_items[:batch_size]
    robots = list(available_robots)

    # Build cost matrix using A* path lengths.
    cost = np.zeros((len(robots), len(items)), dtype=np.float64)
    for i, r in enumerate(robots):
        for j, it in enumerate(items):
            cost[i, j] = _path_distance(env, r, it) + 1e-6 * j  # stable FIFO tie-breaker

    # Solve rectangular assignment.
    if cost.shape[0] <= cost.shape[1]:
        row_ind, col_ind = _linear_sum_assignment_min(cost)
        pairs = [(robots[i], items[j]) for i, j in zip(row_ind, col_ind)]
    else:
        # More robots than items: solve transposed and invert (assign each item to one robot).
        row_ind, col_ind = _linear_sum_assignment_min(cost.T)
        pairs = [(robots[j], items[i]) for i, j in zip(row_ind, col_ind)]

    out: List[Assignment] = []
    used_robots = set()
    used_items = set()
    for r, it in pairs:
        if r in used_robots or it in used_items:
            continue
        # Filter out impossible assignments (no-path => huge cost)
        if _path_distance(env, r, it) >= 10**9:
            continue
        out.append(Assignment(robot=r, item=it))
        used_robots.add(r)
        used_items.add(it)
    return out
