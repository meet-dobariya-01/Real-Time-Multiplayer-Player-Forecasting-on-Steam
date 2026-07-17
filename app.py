import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_utils import (
    GAME_DISPLAY_TO_FOLDER,
    HORIZON_HOURS,
    HORIZON_LABELS,
    load_steam_data,
    prepare_game_history,
    get_game_options,
    _normalize_game_name,
)
from utils.model_utils import (
    MODEL_DESCRIPTIONS,
    MODEL_ORDER,
    build_model_path,
    discover_best_model,
    get_available_models_for_game,
    load_saved_model,
    predict_with_saved_model,
)
from utils.plotting import build_forecast_chart


st.set_page_config(
    page_title="GamePulse",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as handle:
            st.markdown(f"<style>{handle.read()}</style>", unsafe_allow_html=True)


load_css()


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> Tuple[pd.DataFrame, List[str]]:
    data = load_steam_data()
    options = get_game_options(data)
    return data, options


def render_sidebar(data: pd.DataFrame) -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("## 🎮 GamePulse")
        st.markdown("### Steam Traffic Forecasting")
        st.markdown("---")

        st.markdown("### Select Game")
        game_options = get_game_options(data)
        selected_game = st.selectbox(
            "Choose a Steam title",
            options=game_options,
            index=0,
            label_visibility="visible",
        )

        st.markdown("### Select Forecasting Model")
        available_models = get_available_models_for_game(selected_game)
        # Separate checkbox to use the best model for this game
        use_best_model = st.checkbox(" Use Best Model for this game", value=False)
        if use_best_model:
            selected_model = "⭐ Best Model"
        else:
            model_options = [m for m in MODEL_ORDER if m in available_models]
            if not model_options:
                st.warning("No specific pretrained models available for this game.")
                selected_model = "⭐ Best Model"
            else:
                selected_model = st.selectbox(
                    "Choose a forecasting engine",
                    options=model_options,
                    index=0,
                    label_visibility="visible",
                )

        st.markdown("### Select Forecast Horizon")
        horizon_label = st.selectbox(
            "Prediction horizon",
            options=list(HORIZON_LABELS.keys()),
            index=4,
            label_visibility="visible",
        )
        horizon_hours = HORIZON_HOURS[horizon_label]

        st.markdown("---")
        predict_button = st.button("Predict", use_container_width=True, type="primary")

        return {
            "game": selected_game,
            "model": selected_model,
            "horizon_label": horizon_label,
            "horizon_hours": horizon_hours,
            "predict": predict_button,
        }


def render_home_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>GamePulse</h1>
            <p>Steam Game Player Traffic Forecasting Platform</p>
            <div class="hero-copy">
                Explore future concurrent player traffic using a curated set of pretrained forecasting models.
                Select a game, choose a model, and generate a premium-quality forecast in seconds.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_game_info(data: pd.DataFrame, game: str, model: str, horizon_label: str) -> None:
    history = prepare_game_history(data, game)
    latest = history.iloc[-1]
    peak = int(history.max())
    resolved_game = _normalize_game_name(game, data["title"].dropna().unique().tolist())
    latest_rank = int(data.loc[data["title"] == resolved_game, "rank"].iloc[-1]) if "rank" in data.columns else "N/A"

    info_cards = [
        ("Game Name", game),
        ("Current Rank", f"#{latest_rank}"),
        ("Selected Model", model),
        ("Selected Horizon", horizon_label),
    ]

    cols = st.columns(3)
    for idx, (label, value) in enumerate(info_cards):
        with cols[idx % 3]:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )


def render_prediction_results(result: Dict[str, Any]) -> None:
    if not result.get("success"):
        st.error(result.get("message", "Prediction could not be completed."))
        return

    st.success("Prediction generated successfully.")

    cols = st.columns([1.3, 1, 1])
    with cols[0]:
        st.markdown(
            f"<div class='big-metric-card'><div class='metric-label'>Predicted Concurrent Players</div><div class='metric-value'>{int(result['forecast'][-1]):,} Players</div></div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Prediction Time</div><div class='metric-value'>{result['prediction_time']}</div></div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Forecast Horizon</div><div class='metric-value'>{result['horizon_label']}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Forecast Overview")
    fig = build_forecast_chart(result["history"], result["forecast"], result["horizon_label"])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    st.markdown("### Smart Insights")
    insight_cols = st.columns(3)
    for idx, insight in enumerate(result["insights"]):
        with insight_cols[idx % 3]:
            st.markdown(
                f"<div class='insight-card'><div class='metric-label'>{insight['title']}</div><div class='metric-value'>{insight['body']}</div></div>",
                unsafe_allow_html=True,
            )


def render_model_explainer() -> None:
    with st.expander("Model descriptions", expanded=False):
        for model_name in MODEL_ORDER:
            description = MODEL_DESCRIPTIONS.get(model_name, "Pretrained forecasting model for player traffic predictions.")
            st.markdown(f"**{model_name}** — {description}")


def main() -> None:
    data, game_options = load_dashboard_data()
    sidebar_config = render_sidebar(data)

    render_home_header()
    st.markdown("---")

    selected_game = sidebar_config["game"]
    selected_model = sidebar_config["model"]
    horizon_label = sidebar_config["horizon_label"]
    horizon_hours = sidebar_config["horizon_hours"]

    if selected_game:
        render_game_info(data, selected_game, selected_model, horizon_label)
        st.markdown("---")

    if sidebar_config["predict"]:
        with st.spinner("Loading model..."):
            try:
                result = predict_with_saved_model(
                    data=data,
                    game_name=selected_game,
                    model_name=selected_model,
                    horizon_hours=horizon_hours,
                    horizon_label=horizon_label,
                )
            except Exception as exc:
                import traceback
                traceback.print_exc()
                result = {
                    "success": False,
                    "message": f"The prediction pipeline could not complete: {exc}",
                }

        render_prediction_results(result)
    else:
        st.info("Select a game and click Predict to generate a forecast.")

    st.markdown("---")
    render_model_explainer()


if __name__ == "__main__":
    main()
