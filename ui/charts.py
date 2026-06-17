"""Plotly chart generation for sand grain analysis results."""

import plotly.express as px
import plotly.graph_objects as go

from core.morphology import GrainStatistics


def create_size_histogram(stats: GrainStatistics) -> go.Figure:
    """Create an equivalent diameter distribution histogram.

    Args:
        stats: Aggregate grain statistics with d_eq_values populated.

    Returns:
        Plotly Figure object.
    """
    fig = px.histogram(
        x=stats.d_eq_values,
        nbins=20,
        title="粒径分布",
        labels={"x": "等效粒径 (d_eq)", "y": "频数"},
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def create_circularity_sphericity_scatter(stats: GrainStatistics) -> go.Figure:
    """Create a circularity vs sphericity scatter plot.

    Args:
        stats: Aggregate grain statistics with circularity_values
               and sphericity_values populated.

    Returns:
        Plotly Figure object.
    """
    fig = px.scatter(
        x=stats.circularity_values,
        y=stats.sphericity_values,
        title="圆度 vs 球度",
        labels={"x": "圆度", "y": "球度"},
        opacity=0.6,
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="black")))
    fig.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def create_zingg_pie_chart(stats: GrainStatistics) -> go.Figure:
    """Create a Zingg classification pie chart.

    Args:
        stats: Aggregate grain statistics with zingg_counts populated.

    Returns:
        Plotly Figure object.
    """
    labels = list(stats.zingg_counts.keys())
    values = list(stats.zingg_counts.values())
    colors = ["#99ff99", "#66b3ff", "#ff9999"]  # 球状(green), 棒状(blue), 片状(red)

    fig = px.pie(
        names=labels,
        values=values,
        title="Zingg分类",
        color_discrete_sequence=colors[:len(labels)],
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
    )
    return fig
