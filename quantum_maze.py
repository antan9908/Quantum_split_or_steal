"""Small Qiskit discrete-time quantum walk for a grid maze."""

from __future__ import annotations

from collections import deque
from math import ceil, log2
from functools import lru_cache
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector


MazeGrid = Sequence[Sequence[int | str]]
DEFAULT_MAZE: tuple[tuple[int | str, ...], ...] = (
    (0, 0, 1, 1, 1, 0, 0),
    (0, 0, 1, 0, 1, 0, 0),
    ("S", 1, 1, 0, 1, 1, 1),
    (0, 0, 1, 1, 1, 0, 1),
    (0, 0, 1, 0, 0, 0, "E"),
)
DEMO_MAZES: dict[str, tuple[tuple[int | str, ...], ...]] = {
    "Quantum Crossroads": DEFAULT_MAZE,
    "Switchback": (
        ("S", 1, 0, 1, 1, 1, 1),
        (0, 1, 0, 1, 0, 0, 1),
        (0, 1, 1, 1, 0, 1, 1),
        (0, 0, 0, 1, 0, 1, 0),
        (1, 1, 1, 1, 1, 1, 0),
        (1, 0, 0, 0, 0, 1, 0),
        (1, 1, 1, 1, 0, 1, "E"),
    ),
    "Long Corridor": (
        ("S", 1, 1, 0, 1, 1, 1, 1),
        (0, 0, 1, 0, 1, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 0, 0, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 0, 1),
        (0, 0, 0, 0, 0, 1, 1, "E"),
    ),
}
DEFAULT_DEMO_NAME = "Quantum Crossroads"

DIRECTIONS = ("Up", "Right", "Down", "Left")
DIRECTION_STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPPOSITE_DIRECTION = (2, 3, 0, 1)


def validate_maze(maze_grid: MazeGrid) -> tuple[tuple[int | str, ...], ...]:
    """Validate and normalize a rectangular grid containing one S and one E."""
    maze = tuple(tuple(row) for row in maze_grid)
    if not maze or not maze[0] or any(len(row) != len(maze[0]) for row in maze):
        raise ValueError("The maze must be a non-empty rectangular grid.")
    if any(cell not in (0, 1, "S", "E") for row in maze for cell in row):
        raise ValueError("Maze cells must be 0 (wall), 1 (path), 'S', or 'E'.")
    if sum(cell == "S" for row in maze for cell in row) != 1:
        raise ValueError("The maze must contain exactly one Start ('S').")
    if sum(cell == "E" for row in maze for cell in row) != 1:
        raise ValueError("The maze must contain exactly one Exit ('E').")
    return maze


def maze_points(maze_grid: MazeGrid) -> tuple[tuple[int, int], tuple[int, int]]:
    maze = validate_maze(maze_grid)
    start = next((r, c) for r, row in enumerate(maze) for c, cell in enumerate(row) if cell == "S")
    exit_point = next((r, c) for r, row in enumerate(maze) for c, cell in enumerate(row) if cell == "E")
    return start, exit_point


def shortest_maze_path(
    maze_grid: MazeGrid,
) -> list[tuple[int, int]]:
    """Return one classical shortest path through open cells, including S and E."""
    maze = validate_maze(maze_grid)
    start, exit_point = maze_points(maze)
    queue = deque([start])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        row, column = queue.popleft()
        if (row, column) == exit_point:
            break
        for dr, dc in DIRECTION_STEPS:
            neighbor = (row + dr, column + dc)
            nr, nc = neighbor
            if (
                0 <= nr < len(maze)
                and 0 <= nc < len(maze[0])
                and maze[nr][nc] != 0
                and neighbor not in previous
            ):
                previous[neighbor] = (row, column)
                queue.append(neighbor)

    if exit_point not in previous:
        return []
    path = []
    current: tuple[int, int] | None = exit_point
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))


def _position_layout(maze: tuple[tuple[int | str, ...], ...]) -> tuple[int, int, int, int]:
    height, width = len(maze), len(maze[0])
    x_bits = max(1, ceil(log2(width)))
    y_bits = max(1, ceil(log2(height)))
    return height, width, x_bits, y_bits


def _basis_index(row: int, column: int, direction: int, x_bits: int) -> int:
    """Qiskit little-endian layout: coin, x/column, then y/row."""
    return direction + (column << 2) + (row << (2 + x_bits))


def create_initial_state(maze_grid: MazeGrid = DEFAULT_MAZE) -> np.ndarray:
    """Start at S with probability one and the direction coin in |00> (Up)."""
    maze = validate_maze(maze_grid)
    start, _ = maze_points(maze)
    height, width, x_bits, y_bits = _position_layout(maze)
    state = np.zeros(1 << (2 + x_bits + y_bits), dtype=complex)
    state[_basis_index(*start, 0, x_bits)] = 1.0
    return state


@lru_cache(maxsize=16)
def _reflecting_shift(maze: tuple[tuple[int | str, ...], ...]) -> np.ndarray:
    """Build a reversible position-and-direction permutation for the maze."""
    height, width, x_bits, y_bits = _position_layout(maze)
    qubit_count = 2 + x_bits + y_bits
    dimension = 1 << qubit_count
    shift = np.zeros((dimension, dimension), dtype=complex)
    x_mask = (1 << x_bits) - 1
    y_mask = (1 << y_bits) - 1

    for source in range(dimension):
        direction = source & 0b11
        column = (source >> 2) & x_mask
        row = (source >> (2 + x_bits)) & y_mask

        # Padded coordinates and walls are unreachable from a valid start;
        # identity keeps the full operation unitary on those basis states.
        if row >= height or column >= width or maze[row][column] == 0:
            target = source
        else:
            dr, dc = DIRECTION_STEPS[direction]
            next_row, next_column = row + dr, column + dc
            can_move = (
                0 <= next_row < height
                and 0 <= next_column < width
                and maze[next_row][next_column] != 0
            )
            if can_move:
                target = _basis_index(next_row, next_column, direction, x_bits)
            else:
                target = _basis_index(
                    row, column, OPPOSITE_DIRECTION[direction], x_bits
                )
        shift[target, source] = 1.0

    # A permutation matrix is unitary only if every output has one preimage.
    if not np.allclose(np.sum(shift, axis=0), 1.0) or not np.allclose(
        np.sum(shift, axis=1), 1.0
    ):
        raise ValueError("Maze reflection did not produce a reversible walk step.")
    return shift


def run_quantum_step(
    current_state_vector: np.ndarray,
    maze_grid: MazeGrid = DEFAULT_MAZE,
) -> np.ndarray:
    """Apply H⊗H to the four-direction coin, then shift/reflect through the maze."""
    maze = validate_maze(maze_grid)
    shift = _reflecting_shift(maze)
    expected_dimension = shift.shape[0]
    state = np.asarray(current_state_vector, dtype=complex)
    if state.shape != (expected_dimension,):
        raise ValueError(
            f"State vector must contain {expected_dimension} amplitudes for this maze."
        )
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, atol=1e-8):
        raise ValueError("The input state vector must be normalized.")

    qubit_count = int(log2(expected_dimension))
    circuit = QuantumCircuit(qubit_count)
    circuit.h(0)
    circuit.h(1)
    circuit.append(UnitaryGate(shift, label="Reflecting maze shift"), range(qubit_count))
    return np.asarray(Statevector(state).evolve(circuit).data, dtype=complex)


def position_probabilities(
    state_vector: np.ndarray,
    maze_grid: MazeGrid = DEFAULT_MAZE,
) -> dict[tuple[int, int], float]:
    """Sum over direction-coin states to get the probability at each open cell."""
    maze = validate_maze(maze_grid)
    height, width, x_bits, _ = _position_layout(maze)
    amplitudes = np.asarray(state_vector, dtype=complex)
    expected_dimension = 1 << (2 + x_bits + _position_layout(maze)[3])
    if amplitudes.shape != (expected_dimension,):
        raise ValueError("State vector size does not match the maze dimensions.")

    result: dict[tuple[int, int], float] = {}
    for row in range(height):
        for column in range(width):
            if maze[row][column] == 0:
                continue
            result[(row, column)] = float(
                sum(
                    abs(amplitudes[_basis_index(row, column, direction, x_bits)]) ** 2
                    for direction in range(4)
                )
            )
    return result


def coin_probabilities(
    state_vector: np.ndarray,
    maze_grid: MazeGrid = DEFAULT_MAZE,
) -> dict[str, float]:
    """Return the four direction probabilities after summing over positions."""
    maze = validate_maze(maze_grid)
    height, width, x_bits, _ = _position_layout(maze)
    amplitudes = np.asarray(state_vector, dtype=complex)
    probabilities = {direction: 0.0 for direction in DIRECTIONS}
    for row in range(height):
        for column in range(width):
            if maze[row][column] == 0:
                continue
            for direction, name in enumerate(DIRECTIONS):
                index = _basis_index(row, column, direction, x_bits)
                probabilities[name] += float(abs(amplitudes[index]) ** 2)
    return probabilities


def dominant_state_components(
    state_vector: np.ndarray,
    maze_grid: MazeGrid = DEFAULT_MAZE,
    limit: int = 8,
) -> list[tuple[int, int, str, complex]]:
    """Return the largest basis amplitudes as (row, column, direction, amplitude)."""
    maze = validate_maze(maze_grid)
    height, width, x_bits, y_bits = _position_layout(maze)
    amplitudes = np.asarray(state_vector, dtype=complex)
    components = []
    for index, amplitude in enumerate(amplitudes):
        if abs(amplitude) < 1e-8:
            continue
        direction = index & 0b11
        column = (index >> 2) & ((1 << x_bits) - 1)
        row = (index >> (2 + x_bits)) & ((1 << y_bits) - 1)
        if row < height and column < width and maze[row][column] != 0:
            components.append((row, column, DIRECTIONS[direction], complex(amplitude)))
    return sorted(components, key=lambda component: abs(component[3]) ** 2, reverse=True)[:limit]


def draw_maze(
    maze_grid: MazeGrid = DEFAULT_MAZE,
    positions_and_probabilities: dict[tuple[int, int], float] | None = None,
    *,
    ax=None,
):
    """Draw the retro grid and overlay a soft plasma-colored probability field."""
    maze = validate_maze(maze_grid)
    height, width = len(maze), len(maze[0])
    start, exit_point = maze_points(maze)
    base = np.array(
        [[0 if cell == 0 else 1 for cell in row] for row in maze], dtype=float
    )
    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 5.5))
    else:
        figure = ax.figure
    figure.patch.set_facecolor("#080b14")
    ax.set_facecolor("#080b14")
    ax.imshow(base, cmap=ListedColormap(["#080b14", "#d7dce5"]), vmin=0, vmax=1)

    probability_values = positions_and_probabilities or {}
    heat = np.zeros((height, width), dtype=float)
    for (row, column), probability in probability_values.items():
        if 0 <= row < height and 0 <= column < width and maze[row][column] != 0:
            heat[row, column] = max(0.0, float(probability))
    maximum = float(heat.max())
    if maximum > 0:
        alpha = np.where(heat > 0, 0.25 + 0.72 * np.sqrt(heat / maximum), 0.0)
        ax.imshow(heat, cmap="plasma", vmin=0, vmax=maximum, alpha=alpha)
        colorbar = figure.colorbar(
            ScalarMappable(norm=Normalize(vmin=0, vmax=maximum), cmap="plasma"),
            ax=ax,
            fraction=0.035,
            pad=0.025,
        )
        colorbar.set_label("Position probability", color="#cbd5e1")
        colorbar.ax.tick_params(colors="#cbd5e1")

    ax.scatter(start[1], start[0], s=245, marker="o", c="#39ff88",
               edgecolors="#d7ffe7", linewidths=1.3, zorder=5)
    ax.scatter(exit_point[1], exit_point[0], s=320, marker="*", c="#ff4d73",
               edgecolors="#ffe3ea", linewidths=1.0, zorder=5)
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#64748b", linewidth=1.2, alpha=0.38)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    figure.tight_layout(pad=0.25)
    return figure


def update_grid(
    positions_and_probabilities: dict[tuple[int, int], float],
    maze_grid: MazeGrid = DEFAULT_MAZE,
    *,
    ax=None,
):
    """Update the maze view with probability values keyed by (row, column)."""
    return draw_maze(maze_grid, positions_and_probabilities, ax=ax)
