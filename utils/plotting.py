import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go


def build_forecast_chart(history: pd.Series, forecast: pd.Series, horizon_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="Historical Player Traffic", line=dict(color="#4f8cff", width=3)))
    if hasattr(forecast, "__len__"):
        future_index = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=len(forecast), freq="h")
        fig.add_trace(go.Scatter(x=future_index, y=forecast, mode="lines", name="Forecast", line=dict(color="#ff6b6b", width=3, dash="dash")))
        fig.add_vrect(x0=future_index[0], x1=future_index[-1], fillcolor="rgba(255, 107, 107, 0.14)", opacity=0.35, line_width=0)
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=f"Historical vs Forecasted Traffic · {horizon_label}",
    )
    return fig
