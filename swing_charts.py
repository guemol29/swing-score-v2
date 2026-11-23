"""Altair charts for Swing Score V2 technical pattern validation."""
from __future__ import annotations

import altair as alt
import pandas as pd

from swing_patterns import build_pattern_overlays


def build_v2_altair_chart(chart_data: pd.DataFrame, result: dict):
    """Layer price, MM200, base zone, pattern levels and event markers."""
    pattern = {
        "base_start": result.get("v2_base_start"),
        "base_end": result.get("v2_base_end"),
        "base_support": result.get("v2_base_support"),
        "resistance": result.get("v2_resistance"),
        "support_floor": result.get("v2_support_floor"),
        "breakout_high": result.get("v2_breakout_high"),
        "breakout_date": result.get("v2_breakout_date"),
        "throwback_date": result.get("v2_throwback_date"),
        "failure_date": result.get("v2_failure_date"),
        "gap_date": result.get("v2_gap_date"),
    }
    overlays = build_pattern_overlays(chart_data, pattern)
    plot_data = chart_data.rename_axis("Data").reset_index()
    date_axis = alt.X("Data:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=-35))
    tooltip = [
        alt.Tooltip("Data:T", title="Data"),
        alt.Tooltip("Preu:Q", title="Preu", format=",.2f"),
    ]

    layers = [
        alt.Chart(plot_data).mark_line(color="#42A5F5", strokeWidth=2.3).encode(
            x=date_axis,
            y=alt.Y("Preu:Q", title="Preu", scale=alt.Scale(zero=False)),
            tooltip=tooltip,
        )
    ]

    if "MM200 diària" in plot_data and plot_data["MM200 diària"].notna().any():
        layers.append(
            alt.Chart(plot_data).mark_line(color="#123B67", strokeWidth=2.0).encode(
                x=date_axis,
                y=alt.Y("MM200 diària:Q", title="Preu", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("Data:T", title="Data"),
                    alt.Tooltip("MM200 diària:Q", title="MM200", format=",.2f"),
                ],
            )
        )

    if not overlays["base"].empty:
        layers.append(
            alt.Chart(overlays["base"]).mark_rect(color="#F4B942", opacity=0.20).encode(
                x=alt.X("Inici:T"),
                x2=alt.X2("Final:T"),
                y=alt.Y("Suport:Q"),
                y2=alt.Y2("Resistència:Q"),
                tooltip=[
                    alt.Tooltip("Inici:T", title="Inici base"),
                    alt.Tooltip("Final:T", title="Final base"),
                    alt.Tooltip("Suport:Q", format=",.2f"),
                    alt.Tooltip("Resistència:Q", format=",.2f"),
                ],
            )
        )

    if not overlays["levels"].empty:
        levels = overlays["levels"]
        level_domains = ["Resistència", "Suport d'invalidació", "Màxim breakout"]
        level_colors = ["#E69F00", "#D62728", "#7B61A8"]
        layers.append(
            alt.Chart(levels).mark_rule(strokeDash=[7, 4], strokeWidth=1.6).encode(
                y=alt.Y("Valor:Q", title="Preu", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "Nivell:N",
                    scale=alt.Scale(domain=level_domains, range=level_colors),
                    legend=alt.Legend(title="Nivells tècnics", orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("Nivell:N"),
                    alt.Tooltip("Valor:Q", format=",.2f"),
                ],
            )
        )

    if not overlays["events"].empty:
        events = overlays["events"]
        event_domains = ["Breakout", "Throwback", "Invalidació", "Gap extraordinari"]
        event_colors = ["#E69F00", "#2E8B57", "#D62728", "#CC4C02"]
        event_encoding = alt.Color(
            "Esdeveniment:N",
            scale=alt.Scale(domain=event_domains, range=event_colors),
            legend=alt.Legend(title="Esdeveniments", orient="bottom"),
        )
        layers.append(
            alt.Chart(events).mark_point(filled=True, size=130, stroke="white", strokeWidth=1).encode(
                x=alt.X("Data:T"),
                y=alt.Y("Preu:Q", title="Preu", scale=alt.Scale(zero=False)),
                color=event_encoding,
                shape=alt.Shape("Esdeveniment:N", legend=None),
                tooltip=[
                    alt.Tooltip("Esdeveniment:N"),
                    alt.Tooltip("Data:T"),
                    alt.Tooltip("Preu:Q", format=",.2f"),
                ],
            )
        )
        layers.append(
            alt.Chart(events).mark_text(dy=-14, fontSize=11, fontWeight="bold").encode(
                x=alt.X("Data:T"),
                y=alt.Y("Preu:Q", title="Preu", scale=alt.Scale(zero=False)),
                text=alt.Text("Esdeveniment:N"),
                color=event_encoding,
            )
        )

    return (
        alt.layer(*layers)
        .resolve_scale(y="shared")
        .properties(height=430)
        .interactive()
        .configure_axis(grid=True, labelColor="#78808A", titleColor="#4B5563")
        .configure_legend(labelFontSize=11, titleFontSize=11)
    )




