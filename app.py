"""Streamlit interface for Quantum Split or Steal."""

from __future__ import annotations

import random
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import pandas as pd
import plotly.express as px
import streamlit as st

from quantum_engine import Strategy, build_ewl_circuit, simulate_game


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
GAME_RULES_VERSION = 2

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

        Payoffs are calculated from the exact, noise-free outcome
        probabilities, then multiplied by the selected prize pool.
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
        "**Q = diag(i, −i)**. The outcome payoff matrix is a share of the pool "
        r"\(P\): \(u(00)=(P/2,P/2)\), \(u(01)=(0,P)\), "
        r"\(u(10)=(P,0)\), and \(u(11)=(0,0)\)."
    )
    st.markdown(
        "With the three available choices **S, T, and Q**, (Q, Q) yields the "
        r"Split / Split outcome, so each player receives \(P/2\). If either player "
        "changes unilaterally to S or T while the other keeps Q, that player's "
        "expected winnings fall to zero. Therefore Q/Q is a Nash equilibrium "
        "within this implemented strategy set; the claim is specific to these "
        "gates, measurement mapping, and payoff rules."
    )

if st.session_state.game_history:
    latest = st.session_state.game_history[-1]
    st.markdown("## Latest round")
    result_left, result_right = st.columns([1, 1.2])
    with result_left:
        st.markdown(
            f"""
            <div class="round-card">
              <strong>Round {latest['Round']}</strong><br>
              You played <strong>{STRATEGY_LABELS[latest['Your strategy']]}</strong><br>
              {latest.get('Opponent mode', 'Opponent')} played <strong>{STRATEGY_LABELS[latest['Opponent strategy']]}</strong><br><br>
              Prize pool: <strong>${latest['Prize pool']:.2f}</strong><br>
              Your expected winnings: <strong>${latest['Your payoff']:.2f}</strong><br>
              Opponent expected winnings: <strong>${latest['Opponent payoff']:.2f}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with result_right:
        probabilities = latest["Outcome probabilities"]
        outcomes = ["00", "01", "10", "11"]
        probability_frame = pd.DataFrame(
            {
                "Final state": [OUTCOME_LABELS[outcome] for outcome in outcomes],
                "Probability": [probabilities.get(outcome, 0.0) for outcome in outcomes],
            }
        )
        figure = px.bar(
            probability_frame,
            x="Final state",
            y="Probability",
            text_auto=".1%",
            color="Final state",
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            showlegend=False,
            margin=dict(l=10, r=10, t=15, b=10),
            yaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
            xaxis_title="Measured final state (Alice, Bob)",
        )
        st.plotly_chart(figure, width="stretch")

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
