from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go


def candlestick_chart(df: pd.DataFrame, title="", signal_date=None, entry_date=None, exit_date=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    for n in [20, 50, 200]:
        c = f"MA{n}"
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[c], mode="lines", name=c))
    for date, label, symbol in [(signal_date,"Signal","triangle-up"), (entry_date,"Entry","circle"), (exit_date,"Exit","x")]:
        if date is not None and not pd.isna(date):
            match = df[df["Date"] == pd.Timestamp(date)]
            if not match.empty:
                fig.add_trace(go.Scatter(x=[date], y=[match["Close"].iloc[0]], mode="markers", marker_symbol=symbol, marker_size=12, name=label))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=650)
    return fig


def short_put_chart(
    df: pd.DataFrame,
    title="",
    signal_date=None,
    entry_date=None,
    expiry_date=None,
    strike=None,
    first_touch_date=None,
):
    """Underlying chart for a synthetic short-put trade."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Underlying"
    ))
    for n in [20, 50, 200]:
        c = f"MA{n}"
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[c], mode="lines", name=c))

    if strike is not None:
        fig.add_hline(y=float(strike), line_dash="dash", annotation_text=f"Put Strike {float(strike):.2f}")

    marker_specs = [
        (signal_date, "Signal", "triangle-up"),
        (entry_date, "Put Entry", "circle"),
        (first_touch_date, "First Strike Touch", "triangle-down"),
        (expiry_date, "Expiry", "x"),
    ]
    for date, label, symbol in marker_specs:
        if date is None or pd.isna(date):
            continue
        date = pd.Timestamp(date)
        match = df[df["Date"] == date]
        if not match.empty:
            y = match["Close"].iloc[0]
            fig.add_trace(go.Scatter(
                x=[date], y=[y], mode="markers",
                marker_symbol=symbol, marker_size=12, name=label
            ))

    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=680)
    return fig
