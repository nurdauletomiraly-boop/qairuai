"""Grid search algorithms used in the AI Search Algorithms assignment."""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import count
from math import inf
from typing import Any

State = tuple[int, int]
Grid = list[list[str]]
SearchResult = dict[str, Any]


class GridSearch:
    """Search a 4-connected grid where '#' marks blocked cells."""

  def __init__(
        self,
        grid: Grid,
        start: State = (0, 0),
        goal: State = (5, 5),
    ) -> None:
        if not grid or not grid[0]:
            raise ValueError("grid must not be empty")
        if any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("grid must be rectangular")

   self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.goal = goal
        # Same action order as the assignment: Down, Right, Up, Left.
        self.dirs: tuple[State, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))

   def is_valid(self, row: int, col: int) -> bool:
        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
            and self.grid[row][col] != "#"
        )

  def goal_test(self, state: State) -> bool:
        return state == self.goal

   def neighbors(self, state: State):
        row, col = state
        for dr, dc in self.dirs:
            nxt = (row + dr, col + dc)
            if self.is_valid(*nxt):
                yield nxt

  @staticmethod
    def _result(path: list[State], expanded: int) -> SearchResult:
        return {
            "path": path,
            "cost": len(path) - 1 if path else inf,
            "expanded": expanded,
        }

   def dfs(self) -> SearchResult:
      """Depth-first search; does not guarantee an optimal path."""
        if not self.is_valid(*self.start) or not self.is_valid(*self.goal):
            return self._result([], 0)

   stack = [(self.start, [self.start])]
        visited = {self.start}
        expanded = 0
        while stack:
            current, path = stack.pop()
            expanded += 1
            if self.goal_test(current):
                return self._result(path, expanded)
            # Reverse push order so pop order is Down, Right, Up, Left.
            for nxt in reversed(list(self.neighbors(current))):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append((nxt, path + [nxt]))
        return self._result([], expanded)

   def bfs(self) -> SearchResult:
        """Breadth-first search; optimal when every step has equal cost."""
        if not self.is_valid(*self.start) or not self.is_valid(*self.goal):
            return self._result([], 0)

   queue = deque([(self.start, [self.start])])
        visited = {self.start}
        expanded = 0
        while queue:
            current, path = queue.popleft()
            expanded += 1
            if self.goal_test(current):
                return self._result(path, expanded)
            for nxt in self.neighbors(current):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return self._result([], expanded)

   def ucs(self) -> SearchResult:
        """Uniform-cost search (Dijkstra) for positive step costs."""
        if not self.is_valid(*self.start) or not self.is_valid(*self.goal):
            return self._result([], 0)

  serial = count()
        frontier = [(0, next(serial), self.start, [self.start])]
        best_cost = {self.start: 0}
        expanded = 0
        while frontier:
            cost, _, current, path = heappop(frontier)
            if cost != best_cost.get(current):
                continue
            expanded += 1
            if self.goal_test(current):
                return self._result(path, expanded)
            for nxt in self.neighbors(current):
                new_cost = cost + 1
                if new_cost < best_cost.get(nxt, inf):
                    best_cost[nxt] = new_cost
                    heappush(frontier, (new_cost, next(serial), nxt, path + [nxt]))
        return self._result([], expanded)

   def heuristic(self, state: State) -> int:
        """Manhattan distance to the goal."""
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

  def astar(self) -> SearchResult:
        """A* with Manhattan distance; optimal on this four-direction grid."""
        if not self.is_valid(*self.start) or not self.is_valid(*self.goal):
            return self._result([], 0)

   serial = count()
        h0 = self.heuristic(self.start)
        frontier = [(h0, h0, next(serial), 0, self.start, [self.start])]
        best_cost = {self.start: 0}
        expanded = 0
        while frontier:
            _, _, _, cost, current, path = heappop(frontier)
            if cost != best_cost.get(current):
                continue
            expanded += 1
            if self.goal_test(current):
                return self._result(path, expanded)
            for nxt in self.neighbors(current):
                new_cost = cost + 1
                if new_cost < best_cost.get(nxt, inf):
                    best_cost[nxt] = new_cost
                    h = self.heuristic(nxt)
                    heappush(
                        frontier,
                        (new_cost + h, h, next(serial), new_cost, nxt, path + [nxt]),
                    )
        return self._result([], expanded)
    
