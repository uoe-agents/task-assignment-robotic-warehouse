from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import time


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


def _coords_original_loc_map(env) -> Dict[Tuple[int, int], int]:
    # env.action_id_to_coords_map maps action_id -> (y, x)
    return {v: k for k, v in env.action_id_to_coords_map.items()}


def _non_goal_location_ids(env) -> np.ndarray:
    # non_goal_location_ids corresponds to the item ordering in `get_empty_shelf_information`
    out = []
    for id_, coords in env.action_id_to_coords_map.items():
        if (coords[1], coords[0]) not in env.goals:
            out.append(id_)
    return np.array(out, dtype=int)


def _dist_yx(env, robot, start_yx: Tuple[int, int], goal_yx: Tuple[int, int], cache: Dict) -> int:
    key = (start_yx, goal_yx)
    if key in cache:
        return cache[key]
    path = env.find_path(start_yx, goal_yx, robot, care_for_agents=False)
    if path is None:
        cache[key] = 10**9
    elif len(path) == 0 and start_yx != goal_yx:
        cache[key] = 10**9
    else:
        cache[key] = len(path)
    return cache[key]


def _closest_goal_and_cost(env, robot, from_yx: Tuple[int, int], cache: Dict) -> Tuple[Tuple[int, int], int]:
    # env.goals are (x, y); A* expects (y, x)
    best = None
    best_cost = 10**9
    for (x, y) in env.goals:
        c = _dist_yx(env, robot, from_yx, (y, x), cache)
        if c < best_cost:
            best_cost = c
            best = (y, x)
    return best, best_cost


def _closest_empty_and_cost(
    env,
    robot,
    from_yx: Tuple[int, int],
    empty_location_ids: Sequence[int],
    location_map: Dict[int, Tuple[int, int]],
    cache: Dict,
) -> Tuple[int, Tuple[int, int], int]:
    best_id = -1
    best_yx = None
    best_cost = 10**9
    for loc_id in empty_location_ids:
        yx = location_map[loc_id]
        c = _dist_yx(env, robot, from_yx, yx, cache)
        if c < best_cost:
            best_cost = c
            best_id = loc_id
            best_yx = yx
    return best_id, best_yx, best_cost


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


def allocate_bnb_optimal_sequence(
    env,
    available_robots: Sequence,
    request_queue: Sequence,
    already_assigned_item_ids: Iterable[int],
    batch_size: int | None = None,
    max_tasks_per_robot: int = 2,
    node_budget: int = 100_000,
    max_seconds: float | None = 2.0,
    empty_candidates_k: int = 25,
):
    """
    Centralized optimal MRTA via Branch-and-Bound task-sequence search (small-team baseline).

    We perform a limited-horizon lookahead: consider the first `batch_size` *unassigned* FIFO requests and assign
    them to robots, allowing each robot up to `max_tasks_per_robot` tasks in sequence. Each task uses the same
    deterministic intra-task behavior as the heuristic:
      robot -> shelf (pick) -> closest goal (deliver) -> closest empty shelf (return).

    Objective: minimize total A* distance over the planned tasks (sum over all robots).

    Returns immediate assignments (first task per robot) that can be executed now, while planning may include
    additional tasks for lookahead.
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

    max_tasks_per_robot = max(1, int(max_tasks_per_robot))

    coords_to_loc_id = _coords_original_loc_map(env)
    location_map = env.action_id_to_coords_map
    non_goal_ids = _non_goal_location_ids(env)
    empty_mask = env.get_empty_shelf_information()
    empty_ids_init = set(non_goal_ids[empty_mask > 0].tolist())

    # Dynamic state for search
    start_positions = [(r.y, r.x) for r in robots]
    tasks_count = [0 for _ in robots]
    # Store planned sequences as indices into `items`
    sequences: List[List[int]] = [[] for _ in robots]

    best_cost = float("inf")
    best_sequences: List[List[int]] | None = None
    nodes = 0
    t0 = time.perf_counter()

    # Cache per-robot since their movement constraints are identical for our distance proxy
    dist_cache_per_robot: List[Dict] = [dict() for _ in robots]

    # Precompute candidate return slots per goal to avoid scanning thousands of empties at every node.
    goal_yxs = [(y, x) for (x, y) in env.goals]
    non_goal_loc_ids = [int(i) for i in non_goal_ids.tolist()]
    # Manhattan-sorted rack slot ids for each goal
    goal_to_rack_ids_sorted: Dict[Tuple[int, int], List[int]] = {}
    for goal_yx in goal_yxs:
        gy, gx = goal_yx
        goal_to_rack_ids_sorted[goal_yx] = sorted(
            non_goal_loc_ids,
            key=lambda loc_id: abs(location_map[loc_id][0] - gy) + abs(location_map[loc_id][1] - gx),
        )

    empty_candidates_k = max(1, int(empty_candidates_k))

    def _candidate_empty_ids(goal_yx: Tuple[int, int], empties: set) -> List[int]:
        # Return up to K currently-empty rack slot ids, biased to be close (Manhattan) to the goal.
        out: List[int] = []
        for loc_id in goal_to_rack_ids_sorted.get(goal_yx, []):
            if loc_id in empties:
                out.append(loc_id)
                if len(out) >= empty_candidates_k:
                    break
        return out

    # Simple greedy upper bound (also provides a feasible initial solution)
    def greedy_upper_bound():
        pos = list(start_positions)
        empties = set(empty_ids_init)
        seqs = [[] for _ in robots]
        cost = 0
        for t_idx, item in enumerate(items):
            # pick robot with minimal full-task cost (with current positions)
            best_r = None
            best_inc = 10**9
            best_end_yx = None
            best_return_id = None
            best_pick_loc_id = coords_to_loc_id[(item.y, item.x)]
            for r_i, r in enumerate(robots):
                if len(seqs[r_i]) >= max_tasks_per_robot:
                    continue
                cache = {}
                c1 = _dist_yx(env, r, pos[r_i], (item.y, item.x), cache)
                goal_yx, c2 = _closest_goal_and_cost(env, r, (item.y, item.x), cache)
                if goal_yx is None:
                    continue
                if not empties:
                    continue
                cand = _candidate_empty_ids(goal_yx, empties)
                if not cand:
                    continue
                ret_id, ret_yx, c3 = _closest_empty_and_cost(env, r, goal_yx, cand, location_map, cache)
                inc = c1 + c2 + c3
                if inc < best_inc:
                    best_inc = inc
                    best_r = r_i
                    best_end_yx = ret_yx
                    best_return_id = ret_id
            if best_r is None:
                continue
            seqs[best_r].append(t_idx)
            cost += best_inc
            # update empties approximately: pickup frees its location, return occupies a location
            empties.add(best_pick_loc_id)
            empties.discard(best_return_id)
            pos[best_r] = best_end_yx
        return cost, seqs

    ub_cost, ub_seqs = greedy_upper_bound()
    best_cost = ub_cost
    best_sequences = ub_seqs

    def lower_bound(remaining_items: Sequence, robot_positions: Sequence[Tuple[int, int]]):
        # optimistic: each remaining item cost is min distance from any robot to its shelf (ignore goal/return)
        lb = 0
        for item in remaining_items:
            best = 10**9
            shelf_yx = (item.y, item.x)
            for r_i, r in enumerate(robots):
                c = _dist_yx(env, r, robot_positions[r_i], shelf_yx, dist_cache_per_robot[r_i])
                if c < best:
                    best = c
            lb += best
        return lb

    def dfs(t_i: int, cur_cost: float, robot_positions: List[Tuple[int, int]], empties: set):
        nonlocal best_cost, best_sequences, nodes
        if max_seconds is not None and (time.perf_counter() - t0) > max_seconds:
            return
        nodes += 1
        if nodes > node_budget:
            return
        if t_i >= len(items):
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_sequences = [list(s) for s in sequences]
            return

        # bound
        rem = items[t_i:]
        if cur_cost + lower_bound(rem, robot_positions) >= best_cost:
            return

        item = items[t_i]
        shelf_yx = (item.y, item.x)
        pick_loc_id = coords_to_loc_id[shelf_yx]

        # Branch over which robot takes this task next (this defines per-robot sequence ordering).
        # Explore in increasing estimated incremental cost to find good solutions early.
        candidates = []
        for r_i, r in enumerate(robots):
            if tasks_count[r_i] >= max_tasks_per_robot:
                continue
            cache = dist_cache_per_robot[r_i]
            c1 = _dist_yx(env, r, robot_positions[r_i], shelf_yx, cache)
            goal_yx, c2 = _closest_goal_and_cost(env, r, shelf_yx, cache)
            if goal_yx is None or c1 >= 10**9 or c2 >= 10**9:
                continue
            if not empties:
                continue
            candidate_ids = _candidate_empty_ids(goal_yx, empties)
            if not candidate_ids:
                continue
            ret_id, ret_yx, c3 = _closest_empty_and_cost(env, r, goal_yx, candidate_ids, location_map, cache)
            if ret_id < 0 or ret_yx is None or c3 >= 10**9:
                continue
            inc = c1 + c2 + c3
            candidates.append((inc, r_i, ret_id, ret_yx))

        candidates.sort(key=lambda x: x[0])
        for inc, r_i, ret_id, ret_yx in candidates:
            # apply
            sequences[r_i].append(t_i)
            tasks_count[r_i] += 1
            prev_pos = robot_positions[r_i]
            robot_positions[r_i] = ret_yx
            # update empties: pickup frees its location, return occupies one location
            pick_was_empty_before = pick_loc_id in empties
            empties.add(pick_loc_id)
            # ret_id is chosen from current empties; occupy it
            empties.remove(ret_id)

            dfs(t_i + 1, cur_cost + inc, robot_positions, empties)

            # rollback
            empties.add(ret_id)
            if not pick_was_empty_before:
                empties.remove(pick_loc_id)
            robot_positions[r_i] = prev_pos
            tasks_count[r_i] -= 1
            sequences[r_i].pop()

        # Only skip if no feasible robot can take this item in the current (approximate) state.
        if not candidates:
            dfs(t_i + 1, cur_cost, robot_positions, empties)

    dfs(0, 0.0, list(start_positions), set(empty_ids_init))

    if not best_sequences:
        return []

    # Convert the planned sequences to immediate assignments (first task per robot).
    out: List[Assignment] = []
    used_items = set()
    for r_i, seq in enumerate(best_sequences):
        if not seq:
            continue
        t_idx = seq[0]
        if t_idx in used_items:
            continue
        used_items.add(t_idx)
        out.append(Assignment(robot=robots[r_i], item=items[t_idx]))
    return out
