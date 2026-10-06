"""Modular Eisert-Wilkens-Lewenstein quantum Split or Steal engine.

Outcome keys in this module are always written ``AliceBob``. Qiskit stores
qubit 0 as the least-significant bit; since Alice is qubit 0, statevector
probabilities are explicitly reordered into AliceBob labels here.

At maximal entanglement the EWL operator is
J = exp(i*pi/4 * X⊗X) = (I⊗I + i X⊗X) / sqrt(2).
The coefficient must be 1/sqrt(2) for J to be unitary.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import RXXGate, UnitaryGate
from qiskit.quantum_info import Statevector


Strategy = Literal["S", "T", "Q"]

# Keys use AliceBob order. Values are shares of the prize pool.
PAYOFF_MATRIX: dict[str, tuple[float, float]] = {
    "00": (0.5, 0.5),  # Split, Split: divide the prize equally
    "01": (0.0, 1.0),  # Alice splits; Bob steals
    "10": (1.0, 0.0),  # Alice steals; Bob splits
    "11": (0.0, 0.0),  # Steal, Steal: nobody receives the prize
}

def _strategy_gate(strategy: Strategy) -> Gate:
    """Return the single-qubit gate for Split, Steal, or Quantum."""
    if strategy == "S":
        return QuantumCircuit(1, name="Split").to_gate()
    if strategy == "T":
        circuit = QuantumCircuit(1, name="Steal")
        circuit.x(0)
        return circuit.to_gate()
    if strategy == "Q":
        # U_Q = diag(i, -i), equivalent up to global phase to Pauli-Z.
        return UnitaryGate(np.diag([1j, -1j]), label="Q")
    raise ValueError(f"Unknown strategy {strategy!r}; expected 'S', 'T', or 'Q'.")


def build_ewl_circuit(
    alice_strategy: Strategy = "S",
    bob_strategy: Strategy = "S",
    *,
    measure: bool = True,
) -> QuantumCircuit:
    """Build the maximal-entanglement EWL circuit for two player strategies.

    Qubit 0 belongs to Alice and qubit 1 to Bob. Set ``measure=False`` for
    statevector calculations; when measurements are requested, classical bit
    0 is Alice and classical bit 1 is Bob. Split is identity, Steal is Pauli-X.
    """
    alice_gate = _strategy_gate(alice_strategy)
    bob_gate = _strategy_gate(bob_strategy)

    circuit = QuantumCircuit(2, 2 if measure else 0)
    # RXX(theta) = exp(-i*theta/2 * X⊗X), so theta=-pi/2 gives J.
    circuit.append(RXXGate(-pi / 2, label="J"), [0, 1])
    circuit.append(alice_gate, [0])
    circuit.append(bob_gate, [1])
    circuit.append(RXXGate(pi / 2, label="J†"), [0, 1])

    if measure:
        circuit.measure(0, 0)  # Alice -> classical bit 0
        circuit.measure(1, 1)  # Bob -> classical bit 1
    return circuit


def outcome_probabilities(
    alice_strategy: Strategy = "S",
    bob_strategy: Strategy = "S",
) -> dict[str, float]:
    """Return exact, noise-free probabilities with keys in AliceBob order."""
    circuit = build_ewl_circuit(alice_strategy, bob_strategy, measure=False)
    state = Statevector.from_instruction(circuit)
    raw_probabilities = state.probabilities()

    probabilities: dict[str, float] = {}
    for index, probability in enumerate(raw_probabilities):
        # Qubit 0 is the least-significant statevector bit (Alice).
        alice_bit = index & 1
        bob_bit = (index >> 1) & 1
        outcome = f"{alice_bit}{bob_bit}"
        if probability > 0.0:
            probabilities[outcome] = float(probability)
    return probabilities


def expected_payoffs(
    probabilities: dict[str, float],
    payoff_matrix: dict[str, tuple[float, float]] | None = None,
    prize_amount: float = 1.0,
) -> tuple[float, float]:
    """Convert outcome probabilities to expected cash winnings.

    The matrix stores shares of the prize pool; ``prize_amount`` scales those
    shares into the currency units chosen by the caller.
    """
    if prize_amount < 0:
        raise ValueError("prize_amount must be non-negative.")
    matrix = PAYOFF_MATRIX if payoff_matrix is None else payoff_matrix
    alice_payoff = prize_amount * sum(
        matrix[outcome][0] * probability for outcome, probability in probabilities.items()
    )
    bob_payoff = prize_amount * sum(
        matrix[outcome][1] * probability for outcome, probability in probabilities.items()
    )
    return alice_payoff, bob_payoff


def simulate_game(
    alice_strategy: Strategy = "S",
    bob_strategy: Strategy = "S",
    payoff_matrix: dict[str, tuple[float, float]] | None = None,
    prize_amount: float = 1.0,
) -> dict[str, object]:
    """Return the circuit, exact outcome probabilities, and expected payoffs."""
    probabilities = outcome_probabilities(alice_strategy, bob_strategy)
    return {
        "circuit": build_ewl_circuit(alice_strategy, bob_strategy),
        "probabilities": probabilities,
        "expected_payoffs": expected_payoffs(probabilities, payoff_matrix, prize_amount),
    }
