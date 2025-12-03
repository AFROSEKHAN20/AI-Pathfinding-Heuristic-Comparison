from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Iterable, Optional, Callable
import heapq
import math
import time
import os

# Optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


Coord = Tuple[int, int]  # (row, col)


# ============================================================================
# GRID MAP CLASS
# ============================================================================

class GridMap:
    """
    Weighted grid-based map.
    - cost <= 0  => obstacle
    - cost > 0   => traversable with weight
    """

    def __init__(self, width: int, height: int, default_cost: float = 1.0):
        self.width = width
        self.height = height
        self._cells = [[default_cost for _ in range(width)] for _ in range(height)]

    def in_bounds(self, coord: Coord) -> bool:
        r, c = coord
        return 0 <= r < self.height and 0 <= c < self.width

    def is_blocked(self, coord: Coord) -> bool:
        r, c = coord
        return self._cells[r][c] <= 0

    def set_cost(self, coord: Coord, cost: float):
        if not self.in_bounds(coord):
            raise ValueError(f"Coordinate {coord} is out of bounds.")
        r, c = coord
        self._cells[r][c] = cost

    def get_cost(self, coord: Coord) -> float:
        if not self.in_bounds(coord):
            raise ValueError(f"Coordinate {coord} is out of bounds.")
        r, c = coord
        return self._cells[r][c]

    def neighbors_4(self, coord: Coord) -> Iterable[Coord]:
        """Up, Down, Left, Right"""
        r, c = coord
        for n in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if self.in_bounds(n) and not self.is_blocked(n):
                yield n

    def neighbors_8(self, coord: Coord) -> Iterable[Coord]:
        """8-directional neighbors including diagonals."""
        r, c = coord
        for n in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1),
                  (r - 1, c - 1), (r - 1, c + 1),
                  (r + 1, c - 1), (r + 1, c + 1)]:
            if self.in_bounds(n) and not self.is_blocked(n):
                yield n


# ============================================================================
# HEURISTICS
# ============================================================================

def manhattan(a: Coord, b: Coord) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a: Coord, b: Coord) -> float:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


HEURISTICS: Dict[str, Callable[[Coord, Coord], float]] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
}


def get_heuristic(name: str):
    name = name.lower()
    if name not in HEURISTICS:
        raise ValueError(f"Unknown heuristic '{name}'.")
    return HEURISTICS[name]


# ============================================================================
# PRIORITY QUEUE NODE FOR A*
# ============================================================================

@dataclass(order=True)
class PriorityQueueNode:
    f: float
    h: float
    coord: Coord


# ============================================================================
# A* RESULT
# ============================================================================

class AStarResult:
    def __init__(
        self,
        path: List[Coord],
        success: bool,
        nodes_expanded: int,
        execution_time_sec: float,
        total_cost: Optional[float],
    ):
        self.path = path
        self.success = success
        self.nodes_expanded = nodes_expanded
        self.execution_time_sec = execution_time_sec
        self.total_cost = total_cost

    def __str__(self):
        if not self.success:
            return f"FAILED | Expanded={self.nodes_expanded} | Time={self.execution_time_sec:.6f}s"
        return (
            f"SUCCESS | PathLen={len(self.path)} | Cost={self.total_cost:.2f} | "
            f"Expanded={self.nodes_expanded} | Time={self.execution_time_sec:.6f}s"
        )


# ============================================================================
# A* SEARCH
# ============================================================================

def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star_search(
    grid: GridMap,
    start: Coord,
    goal: Coord,
    heuristic_name: str,
    use_diagonal: bool = True,
) -> AStarResult:

    h_func = get_heuristic(heuristic_name)

    if grid.is_blocked(start) or grid.is_blocked(goal):
        raise ValueError("Start or goal is blocked.")

    neighbors_fn = grid.neighbors_8 if use_diagonal else grid.neighbors_4

    open_set: List[PriorityQueueNode] = []
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}

    start_h = h_func(start, goal)
    heapq.heappush(open_set, PriorityQueueNode(start_h, start_h, start))

    closed_set = set()
    expanded = 0

    t0 = time.perf_counter()

    while open_set:
        current_node = heapq.heappop(open_set)
        current = current_node.coord

        if current in closed_set:
            continue

        if current == goal:
            t1 = time.perf_counter()
            return AStarResult(
                reconstruct_path(came_from, current),
                True,
                expanded,
                t1 - t0,
                g_score[current],
            )

        closed_set.add(current)
        expanded += 1

        for nbr in neighbors_fn(current):
            tentative_g = g_score[current] + grid.get_cost(nbr)

            if nbr in g_score and tentative_g >= g_score[nbr]:
                continue

            came_from[nbr] = current
            g_score[nbr] = tentative_g
            h_val = h_func(nbr, goal)
            f_val = tentative_g + h_val
            heapq.heappush(open_set, PriorityQueueNode(f_val, h_val, nbr))

    t1 = time.perf_counter()
    return AStarResult([], False, expanded, t1 - t0, None)


# ============================================================================
# MAP DEFINITIONS
# ============================================================================

def make_test_map_1():
    gm = GridMap(20, 10, 1.0)
    for c in range(20):
        gm.set_cost((5, c), 0)
    gm.set_cost((5, 10), 1)
    return gm, (0, 0), (9, 19)


def make_test_map_2():
    gm = GridMap(30, 15, 1.0)
    for r in range(4, 11):
        for c in range(8, 22):
            gm.set_cost((r, c), 5.0)

    for r in range(3, 12):
        gm.set_cost((r, 5), 0)
        gm.set_cost((r, 24), 0)

    return gm, (0, 0), (14, 29)


def make_test_map_3():
    gm = GridMap(25, 25, 1.0)
    for c in range(2, 23, 4):
        for r in range(25):
            gm.set_cost((r, c), 0)
        gm.set_cost(((c * 3) % 25, c), 1)
    return gm, (0, 0), (24, 24)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_path(grid, path, start, goal, title="A* Path Visualization"):

    if not HAS_MATPLOTLIB:
        print("Matplotlib not installed. Visualization skipped.")
        return

    height, width = grid.height, grid.width
    data = np.zeros((height, width))

    for r in range(height):
        for c in range(width):
            data[r, c] = -1 if grid.get_cost((r, c)) <= 0 else grid.get_cost((r, c))

    plt.figure(figsize=(6, 6))
    cmap = plt.cm.viridis
    cmap.set_under("black")

    plt.imshow(data, cmap=cmap, interpolation="nearest", vmin=0.0001)
    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    plt.plot(xs, ys, marker="o", markersize=4, linewidth=2, color="cyan")

    plt.scatter(start[1], start[0], s=100, marker="s", color="blue", label="Start")
    plt.scatter(goal[1], goal[0], s=100, marker="X", color="red", label="Goal")

    plt.title(title)
    plt.gca().invert_yaxis()
    plt.colorbar(label="Cell cost (obstacles in black)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    filename = f"images/{title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved visualization → {filename}")

    plt.show()


# ============================================================================
# SUMMARY EXPORT
# ============================================================================

def get_summary_text(results: List[Dict]) -> str:
    lines = []
    header = (
        f"{'Map':<22} {'Heuristic':<11} {'Success':<7} "
        f"{'PathLen':<7} {'Cost':<10} {'NodesExpd':<10} {'Time(s)':<9}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        cost_str = f"{r['total_cost']:.2f}" if r["total_cost"] else "-"
        lines.append(
            f"{r['map']:<22} {r['heuristic']:<11} {str(r['success']):<7} "
            f"{r['path_length']:<7} {cost_str:<10} "
            f"{r['nodes_expanded']:<10} {r['time_sec']:.6f}"
        )

    return "\n".join(lines)


# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

def run_experiments(visualize=False):

    maps = [
        ("Map1_WallGap", make_test_map_1),
        ("Map2_WeightedTerrain", make_test_map_2),
        ("Map3_MazeBars", make_test_map_3),
    ]
    heuristics = ["manhattan", "euclidean", "chebyshev"]
    results = []

    for map_name, fn in maps:
        gm, start, goal = fn()
        print(f"\n=== Running on {map_name} | start={start}, goal={goal} ===")

        for h in heuristics:
            print(f"  Heuristic: {h}")
            res = a_star_search(gm, start, goal, h)
            print("   ", res)

            results.append({
                "map": map_name,
                "heuristic": h,
                "success": res.success,
                "path_length": len(res.path),
                "total_cost": res.total_cost,
                "nodes_expanded": res.nodes_expanded,
                "time_sec": res.execution_time_sec,
            })

            if visualize and res.success and h == "manhattan":
                visualize_path(
                    gm, res.path, start, goal,
                    f"{map_name} - {h} heuristic"
                )

    print("\n===== COMPARATIVE PERFORMANCE SUMMARY =====")
    summary = get_summary_text(results)
    print(summary)

    os.makedirs("results", exist_ok=True)
    with open("results/comparison_output.txt", "w") as f:
        f.write(summary)

    print("\nSaved comparison summary → results/comparison_output.txt")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_experiments(visualize=True)   # Change to False to disable visualization
