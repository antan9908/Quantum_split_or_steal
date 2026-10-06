# Quantum Split or Steal
## Live Demo

[Play Quantum Split or Steal](https://quantumsplitorsteal-92swqsx6hmenzt9wdt8s4f.streamlit.app/)
An interactive **Eisert–Wilkens–Lewenstein (EWL) quantum game** built with Qiskit and Streamlit. Choose **Split**, **Steal**, or **Quantum**, play against a random AI or a manually selected opponent, and see the exact outcome probabilities, circuit, and winnings for each round.

## Game rules

Choose a prize pool for each round. The classical Split or Steal payoff is:

| Your move | Opponent's move | Your winnings | Opponent winnings |
| --- | --- | ---: | ---: |
| Split | Split | Half the prize | Half the prize |
| Split | Steal | $0 | Full prize |
| Steal | Split | Full prize | $0 |
| Steal | Steal | $0 | $0 |

The quantum option applies an EWL phase operation. The app calculates exact, noise-free probabilities using Qiskit's `Statevector`, then calculates payouts from the selected prize pool. This project also has a custom rule: when one player chooses **Quantum (Q)** and the other chooses **Split (S)**, each receives half the prize, regardless of the measured outcome. This applies in either player order.

## Strategies and quantum model

- **Split (S):** identity operation.
- **Steal (T):** Pauli-X operation.
- **Quantum (Q):** `diag(i, -i)`.
- **Entangler:** `J = exp(i π/4 · X⊗X) = (I⊗I + iX⊗X)/√2`; the circuit applies `J†` before measurement.

The displayed bitstrings use Alice–Bob order: `0` means Split and `1` means Steal. Qubit 0 is Alice and qubit 1 is Bob. Outcome charts always show the raw EWL measurement probabilities. The Q/S custom payout is applied separately, so its awarded split can differ from the bitstring shown by the circuit.

The custom Q/S payout is a game-design rule, not a consequence of the EWL protocol. Strategy optimality and equilibrium claims depend on the full payout rules; this app does not assert that Q is universally optimal.

## Features

- Dark, neon-styled Streamlit interface.
- Random AI opponent or manually selected opponent move.
- Adjustable prize pool.
- Exact outcome probabilities, expected winnings, and the custom Q/S payout rule.
- Full-session Plotly timeline showing all four outcome probabilities round by round.
- Matplotlib circuit rendering and Plotly outcome chart.
- Session scoreboards, leaderboard, sortable/filterable round history, and reset control.

## Project structure

```text
.
├── app.py                 # Streamlit UI and session state
├── quantum_engine.py      # EWL circuit, Statevector probabilities, payoffs
├── requirements.txt       # Runtime dependencies
├── .streamlit/config.toml # App theme defaults
└── .gitignore             # Local environments, caches, and secrets
```

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Dependency compatibility

The requirements keep Qiskit on the 1.x line and pair it with a compatible Qiskit Aer range. Streamlit, NumPy, Pandas, Matplotlib, and Plotly provide the interface and visualizations.
