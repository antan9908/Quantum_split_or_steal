"""Streamlit interface for Quantum Split or Steal."""

from __future__ import annotations

import random
import os
import time
from io import BytesIO
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import pandas as pd
import plotly.express as px
import streamlit as st

from quantum_engine import PAYOFF_MATRIX, Strategy, build_ewl_circuit, simulate_game
from quantum_maze import (
    DEFAULT_DEMO_NAME,
    DEMO_MAZES,
    DIRECTIONS,
    coin_probabilities,
    create_initial_state,
    dominant_state_components,
    draw_maze,
    maze_points,
    position_probabilities,
    run_quantum_step,
    shortest_maze_path,
    update_grid,
)


STRATEGY_LABELS = {
    "S": "💰 Split (S)",
    "T": "🪙 Steal (T)",
    "Q": "✨ Quantum (Q)",
}
OUTCOME_LABELS = {
    "00": "∣00⟩  Split / Split",
    "01": "∣01⟩  Split / Steal",
    "10": "∣10⟩  Steal / Split",
    "11": "∣11⟩  Steal / Steal",
}
GAME_RULES_VERSION = 4
EXIT_ARRIVAL_THRESHOLD = 1e-12
AUTO_RUN_SAFETY_LIMIT = 256

st.set_page_config(
    page_title="Quantum Split or Steal",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {color-scheme: dark;}
      html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at 20% -15%, #172554 0%, #0b1020 42%, #070b14 100%);
        color: #e5e7eb; font-family: Inter, "Segoe UI", system-ui, sans-serif;
      }
      [data-testid="stHeader"] {background: rgba(7, 11, 20, .7);}
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px;}
      [data-testid="stSidebar"] {background: linear-gradient(180deg, #0c1427, #090e19); border-right: 1px solid #1e293b;}
      h1, h2, h3 {color: #f8fafc; letter-spacing: -.025em;}
      [data-testid="stMarkdownContainer"] p, label {color: #cbd5e1;}
      .hero {
        padding: 2.1rem 2.3rem; border-radius: 22px; margin-bottom: 1.5rem;
        background: linear-gradient(115deg, rgba(12, 20, 39, .96), rgba(30, 27, 75, .92) 60%, rgba(6, 78, 99, .85));
        color: #fff; border: 1px solid rgba(103, 232, 249, .35);
        box-shadow: 0 0 28px rgba(34, 211, 238, .12), 0 18px 55px rgba(0, 0, 0, .35);
      }
      .hero h1 {font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1.1; margin: 0 0 .6rem 0;}
      .hero p {font-size: 1.05rem; color: #a5f3fc; margin: 0;}
      div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(15, 23, 42, .94), rgba(17, 24, 39, .78));
        border: 1px solid #26354d; padding: 1rem 1.1rem; border-radius: 16px;
        box-shadow: inset 0 1px rgba(255,255,255,.035), 0 8px 24px rgba(0,0,0,.2);
      }
      [data-testid="stMetricValue"] {color: #67e8f9; text-shadow: 0 0 16px rgba(34,211,238,.24);}
      [data-testid="stMetricLabel"] {color: #94a3b8;}
      [data-testid="stBaseButton-primary"] {
        border: 1px solid #22d3ee; color: #06111c;
        background: linear-gradient(100deg, #67e8f9, #a78bfa);
        font-weight: 750; box-shadow: 0 0 22px rgba(34,211,238,.2);
      }
      [data-testid="stBaseButton-secondary"] {border: 1px solid #334155; background: #111827; color: #cbd5e1;}
      [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
      [data-testid="stRadio"] {background-color: rgba(15,23,42,.8);}
      [data-testid="stExpander"] {border: 1px solid #26354d; border-radius: 14px; background: rgba(12,18,32,.72);}
      [data-testid="stDataFrame"] {border: 1px solid #26354d; border-radius: 14px; overflow: hidden;}
      hr {border-color: #26354d;}
      .round-card {
        padding: 1.15rem 1.3rem; border: 1px solid rgba(103,232,249,.25); border-radius: 16px;
        background: linear-gradient(145deg, rgba(15,23,42,.95), rgba(17,24,39,.78));
        box-shadow: 0 0 24px rgba(34,211,238,.07); margin: .5rem 0; line-height: 1.8;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    """Initialize state and discard scores from the previous payoff rules."""
    fresh_state = {
        "rounds_played": 0,
        "player_score": 0.0,
        "opponent_score": 0.0,
        "ai_score": 0.0,
        "game_history": [],
        "player_strategy_choice": "S",
        "opponent_mode_choice": "AI Opponent",
        "opponent_strategy_choice": "S",
        "history_move_filter": ["S", "T", "Q"],
        "history_order": "Newest first",
    }
    if st.session_state.get("game_rules_version") != GAME_RULES_VERSION:
        for key, value in fresh_state.items():
            st.session_state[key] = value
        st.session_state.game_rules_version = GAME_RULES_VERSION
    else:
        for key, value in fresh_state.items():
            if key not in st.session_state:
                st.session_state[key] = value


def reset_game() -> None:
    """Clear game progress and restore controls to their starting choices."""
    fresh_state = {
        "rounds_played": 0,
        "player_score": 0.0,
        "opponent_score": 0.0,
        "ai_score": 0.0,
        "game_history": [],
        "player_strategy_choice": "S",
        "opponent_mode_choice": "AI Opponent",
        "opponent_strategy_choice": "S",
        "history_move_filter": ["S", "T", "Q"],
        "history_order": "Newest first",
    }
    for key, value in fresh_state.items():
        st.session_state[key] = value
    st.session_state.game_rules_version = GAME_RULES_VERSION


def draw_quantum_circuit(circuit):
    """Draw the Qiskit circuit with Matplotlib without global pyplot state."""
    instructions = list(circuit.data)
    figure, axis = plt.subplots(figsize=(max(8, len(instructions) * 1.25), 3.0))
    figure.patch.set_facecolor("#0b1020")
    axis.set_facecolor("#0b1020")
    wire_y = {index: 1 - index for index in range(circuit.num_qubits)}
    right_edge = len(instructions) + 1.0

    for qubit, y in wire_y.items():
        axis.plot([0.0, right_edge], [y, y], color="#475569", linewidth=1.5, zorder=1)
        axis.text(-0.25, y, f"q{qubit}", ha="right", va="center", color="#cbd5e1", fontsize=11)

    for column, item in enumerate(instructions, start=1):
        operation = item.operation
        qubits = [circuit.find_bit(bit).index for bit in item.qubits]
        ys = [wire_y[index] for index in qubits]
        label = operation.label or operation.name.upper()
        if operation.name == "rxx":
            label = operation.label or "RXX"

        if len(ys) > 1:
            axis.plot([column, column], [min(ys), max(ys)], color="#22d3ee", linewidth=2, zorder=2)
        for y in ys:
            box = FancyBboxPatch(
                (column - 0.27, y - 0.17), 0.54, 0.34,
                boxstyle="round,pad=0.04,rounding_size=0.06",
                facecolor="#1e1b4b" if operation.name != "measure" else "#064e3b",
                edgecolor="#a78bfa" if operation.name != "measure" else "#34d399",
                linewidth=1.4,
                zorder=3,
            )
            axis.add_patch(box)
            axis.text(column, y, "M" if operation.name == "measure" else label,
                      ha="center", va="center", color="#f0fdfa", fontsize=9, zorder=4)

    axis.set_xlim(-0.7, right_edge + 0.2)
    axis.set_ylim(-0.65, 1.65)
    axis.axis("off")
    figure.tight_layout()
    return figure


def game_history_explanation(game_history: list[dict]) -> tuple[str, pd.DataFrame]:
    """Explain quantum payouts and list outcome chances for every round."""
    rows = []
    for round_record in game_history:
        prize = round_record["Prize pool"]
        probability_map = round_record["Outcome probabilities"]
        for outcome in ("00", "01", "10", "11"):
            alice_share, bob_share = PAYOFF_MATRIX[outcome]
            rows.append(
                {
                    "Round": round_record["Round"],
                    "Your move": STRATEGY_LABELS[round_record["Your strategy"]],
                    "Opponent move": STRATEGY_LABELS[round_record["Opponent strategy"]],
                    "Possible measured result": OUTCOME_LABELS[outcome],
                    "Chance": probability_map.get(outcome, 0.0) * 100,
                    "Your award for this result": prize * alice_share,
                    "Opponent award for this result": prize * bob_share,
                }
            )

    explanation = (
        "Quantum is a special move, not another name for Split or Steal. "
        "It changes the qubit's phase while the two players' qubits are linked "
        "by the circuit. After the circuit is undone, the app measures both "
        "qubits. That measurement can lead to different Split / Steal results "
        "with the chances shown for every round below. The usual game rules are applied "
        "to each possible result: Split / Split shares the prize, one Steal "
        "takes it all, and Steal / Steal pays nothing.\n\n"
        "The app uses Qiskit's exact, noise-free Statevector calculation, so it "
        "shows the probability-weighted average winnings rather than picking "
        "one random result. This is why choosing Quantum can pay differently "
        "from choosing Split, even against Steal: Quantum changes the chances "
        "of the measured results, while the payout rules for each result stay the same. "
        "The table includes all four possible results for every recorded round; "
        "a 0% chance means that result did not occur in that round's statevector."
    )
    return explanation, pd.DataFrame(rows)


def play_round(
    player_strategy: Strategy,
    opponent_mode: str,
    selected_opponent: Strategy,
    prize_pool: float,
) -> None:
    """Resolve a round and persist its outcomes in the current session."""
    opponent_strategy: Strategy = (
        random.choice(["S", "T", "Q"]) if opponent_mode == "AI Opponent" else selected_opponent
    )
    result = simulate_game(player_strategy, opponent_strategy, prize_amount=prize_pool)
    player_payoff, opponent_payoff = result["expected_payoffs"]

    st.session_state.rounds_played += 1
    st.session_state.player_score += player_payoff
    if opponent_mode == "AI Opponent":
        st.session_state.ai_score += opponent_payoff
    else:
        st.session_state.opponent_score += opponent_payoff
    all_outcomes = {f"{alice}{bob}": 0.0 for alice in "01" for bob in "01"}
    all_outcomes.update(result["probabilities"])
    st.session_state.game_history.append(
        {
            "Round": st.session_state.rounds_played,
            "Opponent mode": opponent_mode,
            "Prize pool": prize_pool,
            "Your strategy": player_strategy,
            "Opponent strategy": opponent_strategy,
            "Your payoff": player_payoff,
            "Opponent payoff": opponent_payoff,
            "Outcome probabilities": all_outcomes,
        }
    )


def render_quantum_maze_tab() -> None:
    """Render the interactive Qiskit maze walk inside the Streamlit app."""
    st.title("Quantum Maze Solver")
    st.caption(
        "A discrete-time quantum walk explores a maze as a probability wave. "
        "Each step uses a Qiskit coin toss followed by a reversible shift that reflects from walls. "
        "Auto-Run continues until the wave first reaches the Exit."
    )

    if "maze_walk_version" not in st.session_state:
        st.session_state.maze_walk_version = 1
        st.session_state.maze_demo_name = DEFAULT_DEMO_NAME
        st.session_state.maze_walk_maze_name = DEFAULT_DEMO_NAME
        st.session_state.maze_walk_state = create_initial_state(DEMO_MAZES[DEFAULT_DEMO_NAME])
        st.session_state.maze_walk_step = 0
        st.session_state.maze_first_arrival_step = None
        st.session_state.maze_first_arrival_probability = None
    if "maze_demo_name" not in st.session_state:
        st.session_state.maze_demo_name = DEFAULT_DEMO_NAME
    if "maze_walk_maze_name" not in st.session_state:
        st.session_state.maze_walk_maze_name = st.session_state.maze_demo_name
    if "maze_first_arrival_step" not in st.session_state:
        st.session_state.maze_first_arrival_step = None
    if "maze_first_arrival_probability" not in st.session_state:
        st.session_state.maze_first_arrival_probability = None

    selected_maze_name = st.selectbox(
        "Choose a demo maze",
        options=list(DEMO_MAZES),
        key="maze_demo_name",
    )
    maze_grid = DEMO_MAZES[selected_maze_name]
    if st.session_state.maze_walk_maze_name != selected_maze_name:
        st.session_state.maze_walk_state = create_initial_state(maze_grid)
        st.session_state.maze_walk_step = 0
        st.session_state.maze_first_arrival_step = None
        st.session_state.maze_first_arrival_probability = None
        st.session_state.maze_walk_maze_name = selected_maze_name

    _, exit_point = maze_points(maze_grid)
    classical_shortest_path = shortest_maze_path(maze_grid)
    state_key = "maze_walk_state"

    def reset_walk() -> None:
        st.session_state[state_key] = create_initial_state(maze_grid)
        st.session_state.maze_walk_step = 0
        st.session_state.maze_first_arrival_step = None
        st.session_state.maze_first_arrival_probability = None

    control_columns = st.columns([1, 1, 1.2, 1.3])
    reset_clicked = control_columns[0].button("↻ Reset", key="maze_reset", width="stretch")
    single_clicked = control_columns[1].button(
        "Step once", key="maze_single_step", width="stretch"
    )
    auto_clicked = control_columns[2].button(
        "▶ Auto-Run to Exit", type="primary", key="maze_auto_run", width="stretch"
    )
    control_columns[3].caption(
        f"Classical shortest route: {len(classical_shortest_path) - 1} steps"
    )

    plot_placeholder = st.empty()
    progress_placeholder = st.empty()
    if reset_clicked:
        reset_walk()
    elif single_clicked:
        st.session_state[state_key] = run_quantum_step(
            st.session_state[state_key], maze_grid
        )
        st.session_state.maze_walk_step += 1
        current_probabilities = position_probabilities(
            st.session_state[state_key], maze_grid
        )
        if (
            st.session_state.maze_first_arrival_step is None
            and current_probabilities.get(exit_point, 0.0) > EXIT_ARRIVAL_THRESHOLD
        ):
            st.session_state.maze_first_arrival_step = st.session_state.maze_walk_step
            st.session_state.maze_first_arrival_probability = current_probabilities[exit_point]
    elif auto_clicked:
        current_probabilities = position_probabilities(
            st.session_state[state_key], maze_grid
        )
        exit_reached = (
            current_probabilities.get(exit_point, 0.0) > EXIT_ARRIVAL_THRESHOLD
        )
        if exit_reached and st.session_state.maze_first_arrival_step is None:
            st.session_state.maze_first_arrival_step = st.session_state.maze_walk_step
            st.session_state.maze_first_arrival_probability = current_probabilities[exit_point]
        for _ in range(0 if exit_reached else AUTO_RUN_SAFETY_LIMIT):
            st.session_state[state_key] = run_quantum_step(
                st.session_state[state_key], maze_grid
            )
            st.session_state.maze_walk_step += 1
            current_probabilities = position_probabilities(
                st.session_state[state_key], maze_grid
            )
            figure = update_grid(current_probabilities, maze_grid)
            plot_placeholder.pyplot(figure, clear_figure=True, width="stretch")
            plt.close(figure)
            progress_placeholder.caption(
                f"Quantum walk advancing: step {st.session_state.maze_walk_step}"
            )
            if current_probabilities.get(exit_point, 0.0) > EXIT_ARRIVAL_THRESHOLD:
                if st.session_state.maze_first_arrival_step is None:
                    st.session_state.maze_first_arrival_step = st.session_state.maze_walk_step
                    st.session_state.maze_first_arrival_probability = current_probabilities[exit_point]
                exit_reached = True
                break
            time.sleep(0.1)
        if not exit_reached:
            st.warning(
                f"The wave did not reach the Exit within {AUTO_RUN_SAFETY_LIMIT} additional steps. "
                "Auto-Run stopped to keep the app responsive; click it again to continue."
            )
        else:
            progress_placeholder.success(
                f"Exit reached by the quantum wave at step {st.session_state.maze_walk_step}."
            )

    state_vector = st.session_state[state_key]
    probabilities = position_probabilities(state_vector, maze_grid)
    figure = draw_maze(maze_grid, probabilities)
    plot_placeholder.pyplot(figure, clear_figure=True, width="stretch")
    image_file = BytesIO()
    figure.savefig(image_file, format="png", facecolor=figure.get_facecolor(), dpi=180)
    plt.close(figure)
    image_file.seek(0)
    st.download_button(
        "Download current maze image",
        data=image_file.getvalue(),
        file_name=(
            f"quantum-maze-{selected_maze_name.lower().replace(' ', '-')}-"
            f"step-{st.session_state.maze_walk_step}.png"
        ),
        mime="image/png",
        key="maze_download_image",
    )

    info_column, coin_column = st.columns([1, 1.2])
    with info_column:
        st.metric("Walk step", st.session_state.maze_walk_step)
        exit_probability = probabilities.get(exit_point, 0.0)
        exit_probability_display = (
            f"{exit_probability:.6%}" if 0 < exit_probability < 0.001
            else f"{exit_probability:.1%}"
        )
        st.metric("Probability at exit", exit_probability_display)
        st.metric("Classical shortest route", f"{len(classical_shortest_path) - 1} steps")
        if st.session_state.maze_first_arrival_step is None:
            st.caption("Quantum wave first arrival: not reached yet")
        else:
            first_arrival_probability = st.session_state.maze_first_arrival_probability
            if first_arrival_probability is None:
                first_arrival_display = "unknown"
            elif first_arrival_probability < 0.001:
                first_arrival_display = f"{first_arrival_probability:.6%}"
            else:
                first_arrival_display = f"{first_arrival_probability:.1%}"
            st.success(
                f"Quantum wave first reached the Exit at step "
                f"{st.session_state.maze_first_arrival_step} "
                f"(Exit probability then: {first_arrival_display})."
            )
        st.caption(
            "The classical shortest route is the fewest adjacent moves from S to E. "
            "Quantum first arrival means the Exit has nonzero probability; a measurement "
            "is not guaranteed to find the walker there."
        )
        peak_cell = max(probabilities, key=probabilities.get)
        st.caption(
            f"Most likely cell: row {peak_cell[0] + 1}, column {peak_cell[1] + 1} "
            f"({probabilities[peak_cell]:.1%})"
        )
    with coin_column:
        st.markdown("#### Direction coin")
        st.caption("The four bars show the chance of each movement direction.")
        coin_frame = pd.DataFrame.from_dict(
            coin_probabilities(state_vector, maze_grid),
            orient="index",
            columns=["Probability"],
        ).reindex(DIRECTIONS)
        st.bar_chart(coin_frame, y="Probability", height=190)

    with st.expander("What is the quantum state right now?"):
        st.markdown(
            "The state vector stores an amplitude for each **position + direction**. "
            "Amplitudes can reinforce or cancel each other; their squared magnitudes "
            "give the probabilities shown on the maze. Here are the largest components:"
        )
        components = dominant_state_components(state_vector, maze_grid)
        for row, column, direction, amplitude in components:
            st.code(
                f"row {row + 1}, column {column + 1}, {direction}: "
                f"{amplitude.real:+.3f}{amplitude.imag:+.3f}i  "
                f"(probability {abs(amplitude) ** 2:.1%})"
            )


selected_app_section = st.segmented_control(
    "Choose app section",
    options=["💰 Split or Steal", "🌀 Quantum Maze Solver"],
    default="💰 Split or Steal",
    label_visibility="collapsed",
    key="app_section",
)

if selected_app_section == "💰 Split or Steal":
    initialize_state()

    with st.sidebar:
        st.markdown("## 📖 How to play")
        st.markdown(
            """
            ### Split or Steal
            Each player secretly chooses **Split** or **Steal**. Let **P** be the
            prize pool for the round:

            | Your move | Other move | You receive | They receive |
            |:--|:--|--:|--:|
            | Split | Split | P / 2 | P / 2 |
            | Split | Steal | 0 | P |
            | Steal | Split | P | 0 |
            | Steal | Steal | 0 | 0 |

            ### Quantum game
            EWL encodes **Split (S)** as identity, **Steal (T)** as Pauli-X, and
            **Quantum (Q)** as a phase operation. After entanglement, the players'
            choices are applied and the qubits are disentangled before measurement.
            The measured bits map to the four Split / Steal outcomes in the table.

            The circuit measures both players' qubits. Those measured results are
            interpreted using the same Split / Steal payout rules above. Quantum
            is a special operation that changes the chances of each result; it is
            not automatically treated as Split or Steal. The app reports expected
            winnings from the exact outcome probabilities.
            """
        )
        st.divider()
        st.markdown("### Move key")
        st.markdown("- **S** — Split\n- **T** — Steal\n- **Q** — Quantum strategy")
        st.divider()
        st.button("↻  Reset Game", on_click=reset_game, width="stretch")

    st.markdown(
        """
        <div class="hero">
          <h1>Quantum Split or Steal</h1>
          <p>Share the prize, claim it all, or explore what a quantum move changes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_mode = st.session_state.get("opponent_mode_choice", "AI Opponent")
    score_col, opponent_score_col, rounds_col = st.columns(3)
    score_col.metric("Your cumulative winnings", f"${st.session_state.player_score:.2f}")
    if current_mode == "AI Opponent":
        opponent_score_col.metric("AI cumulative winnings", f"${st.session_state.ai_score:.2f}")
    else:
        opponent_score_col.metric("Opponent cumulative winnings", f"${st.session_state.opponent_score:.2f}")
    rounds_col.metric("Rounds played", st.session_state.rounds_played)

    st.markdown("## Set up your round")
    player_col, mode_col, opponent_col, prize_col = st.columns([1, 1.15, 1, 0.9])
    with player_col:
        player_choice = st.selectbox(
            "Your move",
            options=["S", "T", "Q"],
            format_func=lambda choice: STRATEGY_LABELS[choice],
            key="player_strategy_choice",
        )
    with mode_col:
        opponent_mode = st.radio(
            "Opponent",
            options=["Manual strategy", "AI Opponent"],
            horizontal=True,
            key="opponent_mode_choice",
        )
    with opponent_col:
        if opponent_mode == "Manual strategy":
            opponent_choice = st.selectbox(
                "Manual opponent move",
                options=["S", "T", "Q"],
                format_func=lambda choice: STRATEGY_LABELS[choice],
                key="opponent_strategy_choice",
            )
        else:
            opponent_choice = "S"
            st.markdown("**AI move**\n\nRandomly chooses Split, Steal, or Quantum each round.")
    with prize_col:
        prize_pool = st.slider("Prize pool ($)", min_value=10, max_value=1000, value=100, step=10)

    if st.button("▶  Play Round", type="primary", width="stretch"):
        play_round(player_choice, opponent_mode, opponent_choice, float(prize_pool))
        st.rerun()

    with st.expander("🧠 Theory & Math — the EWL protocol", expanded=False):
        st.markdown(
            "The Eisert–Wilkens–Lewenstein protocol maps each player's strategy to a "
            "unitary operation on one qubit. It entangles the initial state, applies "
            "the two strategies, then reverses the entanglement before measuring."
        )
        st.latex(r"J(\gamma)=\exp\!\left(i\frac{\gamma}{2}X\otimes X\right),\qquad \gamma=\frac{\pi}{2}")
        st.latex(r"|\psi_f\rangle=J^\dagger(U_A\otimes U_B)J|00\rangle")
        st.latex(r"p_{ab}=|\langle ab|\psi_f\rangle|^2,\qquad \mathbb{E}[u_A]=\sum_{a,b}p_{ab}u_A(ab)")
        st.markdown(
            "For this version, **S (Split) = I**, **T (Steal) = X**, and "
            "**Q = diag(i, −i)**. The measured outcome uses the game's same payout rules: "
            r"\(P\): \(u(00)=(P/2,P/2)\), \(u(01)=(0,P)\), "
            r"\(u(10)=(P,0)\), and \(u(11)=(0,0)\)."
        )
        st.markdown(
            "A Quantum move is not a promise to Split. It changes the quantum state "
            "and therefore the probability of each measured Split / Steal result. "
            "The app applies the ordinary payout table to those results and displays "
            "the expected winnings. Quantum can therefore lead to a different "
            "expected payout than the classical move Split, without changing the "
            "fundamental payout rules."
        )

    if st.session_state.game_history:
        latest = st.session_state.game_history[-1]
        winnings_label = "expected winnings"
        st.markdown("## Latest round")
        st.markdown(
            f"""
            <div class="round-card">
              <strong>Round {latest['Round']}</strong><br>
              You played <strong>{STRATEGY_LABELS[latest['Your strategy']]}</strong><br>
              {latest.get('Opponent mode', 'Opponent')} played <strong>{STRATEGY_LABELS[latest['Opponent strategy']]}</strong><br><br>
              Prize pool: <strong>${latest['Prize pool']:.2f}</strong><br>
              Your {winnings_label}: <strong>${latest['Your payoff']:.2f}</strong><br>
              Opponent {winnings_label}: <strong>${latest['Opponent payoff']:.2f}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        explanation, game_outcomes = game_history_explanation(st.session_state.game_history)
        with st.container(border=True):
            st.markdown("### How Quantum and measured outcomes affect the whole game")
            st.markdown(explanation)
            st.dataframe(
                game_outcomes,
                width="stretch",
                hide_index=True,
                column_config={
                    "Chance": st.column_config.NumberColumn(format="%.1f%%"),
                    "Your award for this result": st.column_config.NumberColumn(format="$%.2f"),
                    "Opponent award for this result": st.column_config.NumberColumn(format="$%.2f"),
                },
            )

        st.markdown("### Outcome probabilities across all rounds")
        st.caption("The exact EWL measurement probabilities for every round. The payout table is applied to these results. Click a legend item to focus on that outcome.")
        outcomes = ["00", "01", "10", "11"]
        probability_history = pd.DataFrame(
            [
                {
                    "Round": round_record["Round"],
                    "Final outcome": OUTCOME_LABELS[outcome],
                    "Probability": round_record["Outcome probabilities"].get(outcome, 0.0),
                }
                for round_record in st.session_state.game_history
                for outcome in outcomes
            ]
        )
        history_figure = px.line(
            probability_history,
            x="Round",
            y="Probability",
            color="Final outcome",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Prism,
            hover_data={"Probability": ":.1%"},
        )
        history_figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            margin=dict(l=10, r=10, t=15, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(title="Round", dtick=1),
            yaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
        )
        st.plotly_chart(history_figure, width="stretch")

        st.markdown("### Quantum circuit")
        circuit = build_ewl_circuit(latest["Your strategy"], latest["Opponent strategy"])
        circuit_figure = draw_quantum_circuit(circuit)
        try:
            st.pyplot(circuit_figure, clear_figure=True, width="stretch")
        finally:
            plt.close(circuit_figure)

        leaderboard_tab, history_tab = st.tabs(["🏆 Leaderboard", "🧾 Round history"])
        with leaderboard_tab:
            st.markdown("### Cumulative standings")
            standings = pd.DataFrame(
                [
                    {"Player": "You", "Cumulative payoff": st.session_state.player_score},
                    {"Player": "AI opponent", "Cumulative payoff": st.session_state.ai_score},
                    {"Player": "Manual opponents", "Cumulative payoff": st.session_state.opponent_score},
                ]
            ).sort_values("Cumulative payoff", ascending=False, ignore_index=True)
            standings.insert(0, "Rank", range(1, len(standings) + 1))
            st.dataframe(
                standings,
                width="stretch",
                hide_index=True,
                column_config={
                    "Cumulative payoff": st.column_config.NumberColumn(format="$%.2f"),
                },
            )
        with history_tab:
            st.markdown("### Round-by-round history")
            history_frame = pd.DataFrame(st.session_state.game_history)
            filter_col, sort_col = st.columns([1, 1])
            with filter_col:
                selected_moves = st.multiselect(
                    "Filter by your move",
                    options=["S", "T", "Q"],
                    format_func=lambda move: STRATEGY_LABELS[move],
                    key="history_move_filter",
                )
            with sort_col:
                history_order = st.selectbox(
                    "Sort rounds",
                    options=["Newest first", "Oldest first", "Highest player payoff"],
                    key="history_order",
                )
            filtered_history = history_frame[history_frame["Your strategy"].isin(selected_moves)]
            if history_order == "Newest first":
                filtered_history = filtered_history.sort_values("Round", ascending=False)
            elif history_order == "Highest player payoff":
                filtered_history = filtered_history.sort_values("Your payoff", ascending=False)
            else:
                filtered_history = filtered_history.sort_values("Round", ascending=True)
            history_display = filtered_history.drop(columns=["Outcome probabilities"]).copy()
            history_display["Your strategy"] = history_display["Your strategy"].map(STRATEGY_LABELS)
            history_display["Opponent strategy"] = history_display["Opponent strategy"].map(STRATEGY_LABELS)
            st.dataframe(
                history_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "Prize pool": st.column_config.NumberColumn(format="$%.2f"),
                    "Your payoff": st.column_config.NumberColumn(format="$%.2f"),
                    "Opponent payoff": st.column_config.NumberColumn(format="$%.2f"),
                },
            )
    else:
        st.caption("Play your first round to see the measured outcome probabilities and game history.")
else:
    render_quantum_maze_tab()
