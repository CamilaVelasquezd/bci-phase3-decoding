"""Interactive Plotly report for the discrete velocity decoder.

Re-runs the pipeline in ``decoding/discrete_decoder.py`` (session 20161021,
DANDI 000688) and renders the results as three interactive figures embedded
in a single static HTML report: per-class accuracy, confusion matrices, and
a colored comparison table.

Run from the repo root:
    PYTHONPATH=/home/camilavelasquez/bci-phase3-decoding python notebooks/discrete_decoding_results.py
"""
from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from decoding.dim_reduction import compute_binned_counts
from decoding.discrete_decoder import (
    apply_pca,
    build_class_names,
    build_models,
    compute_binned_velocity_labels,
    cross_validate_by_trial,
    evaluate_model,
    load_local_session,
    split_by_trial,
)
from decoding.discrete_utils import compute_binned_trial_ids

NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_HTML_PATH = os.path.join(NOTEBOOKS_DIR, "discrete_decoding_results.html")

COLOR_STATIONARY = "#c0005a"
COLOR_SLOW = "#ff6b9d"
COLOR_FAST = "#ffb3d1"
PINK_COLORSCALE = [[0.0, "#ffffff"], [0.5, "#ffb3d1"], [1.0, "#c0005a"]]

TITLE = "Discrete Decoding Results — Session 20161021 (DANDI 000688) — 17 classes"


def run_pipeline() -> dict:
    """Re-run the discrete decoder pipeline end to end.

    Returns
    -------
    dict
        Keys: ``'class_names'`` (list[str], in class-index order present in
        the data), ``'results'`` (dict mapping model name to ``{'val', 'test'}``
        ``evaluate_model`` reports), ``'cv_results'`` (dict mapping model name
        to ``(mean_accuracy, std_accuracy)`` from 5-fold trial-grouped CV).
    """
    bin_size_ms = 50
    ds = load_local_session()

    X_all = compute_binned_counts(ds, bin_size_ms=bin_size_ms)
    y_all = compute_binned_velocity_labels(
        ds, bin_size_ms=bin_size_ms, stat_thresh=0.03, fast_thresh=0.15
    )
    trial_id_binned = compute_binned_trial_ids(ds, bin_size_ms=bin_size_ms)
    assert len(X_all) == len(y_all) == len(trial_id_binned), (
        f"Binned array length mismatch: X={len(X_all)}, y={len(y_all)}, "
        f"trial_id={len(trial_id_binned)}"
    )

    train_mask, val_mask, test_mask = split_by_trial(trial_id_binned)
    X_train, X_val, X_test = X_all[train_mask], X_all[val_mask], X_all[test_mask]
    y_train, y_val, y_test = y_all[train_mask], y_all[val_mask], y_all[test_mask]

    X_train_pca, X_val_pca, X_test_pca, _, _ = apply_pca(X_train, X_val, X_test)

    classes = np.unique(y_all)
    class_names = [build_class_names()[c] for c in classes]

    results = {}
    for name, model in build_models().items():
        model.fit(X_train_pca, y_train)
        results[name] = {
            "val": evaluate_model(model, X_val_pca, y_val, classes, class_names),
            "test": evaluate_model(model, X_test_pca, y_test, classes, class_names),
        }

    cv_results = cross_validate_by_trial(X_all, y_all, trial_id_binned, n_splits=5)

    return {"class_names": class_names, "results": results, "cv_results": cv_results}


def class_group_color(class_name: str) -> str:
    """Map a class name to its speed-group color.

    Parameters
    ----------
    class_name : str
        One of ``'Stationary'``, ``'Slow <dir>'``, or ``'Fast <dir>'``.

    Returns
    -------
    str
        Hex color: dark pink for Stationary, medium pink for Slow, light
        pink for Fast.
    """
    if class_name == "Stationary":
        return COLOR_STATIONARY
    if class_name.startswith("Slow"):
        return COLOR_SLOW
    return COLOR_FAST


def accuracy_to_color(value: float) -> str:
    """Map an accuracy value to a red (low) -> yellow -> green (high) hex color.

    Parameters
    ----------
    value : float
        Accuracy in ``[0, 1]``. NaN (class absent from the split) maps to
        a neutral gray.

    Returns
    -------
    str
        Hex color string.
    """
    if np.isnan(value):
        return "#e0e0e0"
    value = float(np.clip(value, 0.0, 1.0))
    stops = [(0.0, (230, 57, 70)), (0.5, (244, 211, 94)), (1.0, (46, 204, 113))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t0 <= value <= t1:
            frac = 0.0 if t1 == t0 else (value - t0) / (t1 - t0)
            r, g, b = (round(c0[k] + frac * (c1[k] - c0[k])) for k in range(3))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#e0e0e0"


def build_accuracy_bar_figure(results: dict, class_names: list[str]) -> go.Figure:
    """Build a small-multiples horizontal bar chart of per-class test accuracy.

    One subplot per model, bars ordered and colored by speed group
    (Stationary/Slow/Fast), with a dashed reference line at the random
    baseline (``1 / n_classes``).

    Parameters
    ----------
    results : dict
        Output of ``run_pipeline()['results']``.
    class_names : list[str]
        Class names in class-index order.

    Returns
    -------
    go.Figure
    """
    model_names = list(results.keys())
    random_baseline = 1.0 / len(class_names)

    y_order = class_names[::-1]
    colors_order = [class_group_color(c) for c in y_order]

    fig = make_subplots(
        rows=1,
        cols=len(model_names),
        shared_yaxes=True,
        subplot_titles=model_names,
        horizontal_spacing=0.1,
    )

    for col, name in enumerate(model_names, start=1):
        per_class = results[name]["test"]["per_class_accuracy"]
        values = [per_class[c] for c in y_order]
        fig.add_trace(
            go.Bar(
                x=values,
                y=y_order,
                orientation="h",
                marker_color=colors_order,
                showlegend=False,
                hovertemplate="%{y}<br>Accuracy: %{x:.3f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        vline_kwargs = dict(x=random_baseline, line_dash="dash", line_color="#888888")
        if col == 1:
            vline_kwargs["annotation_text"] = f"Random baseline (1/{len(class_names)})"
            vline_kwargs["annotation_position"] = "top"
        fig.add_vline(row=1, col=col, **vline_kwargs)

    for group_name, color in (
        ("Stationary", COLOR_STATIONARY),
        ("Slow", COLOR_SLOW),
        ("Fast", COLOR_FAST),
    ):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color, symbol="square"),
                name=group_name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(range=[0, 1], title_text="Accuracy (test set)")
    fig.update_layout(
        title="Per-Class Accuracy (Test Set)",
        legend_title_text="Speed group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=650,
        width=1150,
    )
    return fig


def build_confusion_matrix_figure(results: dict, class_names: list[str]) -> go.Figure:
    """Build a 2x2 grid of row-normalized confusion matrix heatmaps.

    Layout: (LogisticRegression, val), (LogisticRegression, test),
    (GaussianNB, val), (GaussianNB, test). Color encodes the row-normalized
    (recall) value; hover shows both the normalized value and the raw count.

    Parameters
    ----------
    results : dict
        Output of ``run_pipeline()['results']``.
    class_names : list[str]
        Class names in class-index order, shared by all subplot axes.

    Returns
    -------
    go.Figure
    """
    model_names = list(results.keys())
    splits = ["val", "test"]
    subplot_titles = [f"{model} — {split}" for model in model_names for split in splits]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.15,
        vertical_spacing=0.18,
    )

    for i, model_name in enumerate(model_names, start=1):
        for j, split in enumerate(splits, start=1):
            cm = results[model_name][split]["confusion_matrix"].astype(np.float64)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
            show_colorbar = i == 1 and j == 2

            fig.add_trace(
                go.Heatmap(
                    z=cm_norm,
                    x=class_names,
                    y=class_names,
                    customdata=cm,
                    colorscale=PINK_COLORSCALE,
                    zmin=0,
                    zmax=1,
                    showscale=show_colorbar,
                    colorbar=dict(title="Recall", len=0.4, y=0.8) if show_colorbar else None,
                    hovertemplate=(
                        "True: %{y}<br>Predicted: %{x}<br>Normalized: %{z:.3f}"
                        "<br>Count: %{customdata:.0f}<extra></extra>"
                    ),
                ),
                row=i,
                col=j,
            )

    fig.update_xaxes(tickangle=90, title_text="Predicted")
    fig.update_yaxes(autorange="reversed", title_text="True")
    fig.update_layout(
        title="Confusion Matrices (row-normalized)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=1150,
        width=1200,
    )
    return fig


def build_comparison_table_figure(
    results: dict, class_names: list[str], cv_results: dict
) -> go.Figure:
    """Build a colored comparison table of per-class test accuracy.

    Rows: one per class, plus an overall test accuracy row and a 5-fold
    cross-validation row. Cell backgrounds are colored on a red (low) to
    green (high) scale.

    Parameters
    ----------
    results : dict
        Output of ``run_pipeline()['results']``.
    class_names : list[str]
        Class names in class-index order.
    cv_results : dict
        Output of ``run_pipeline()['cv_results']``: model name ->
        ``(mean_accuracy, std_accuracy)``.

    Returns
    -------
    go.Figure
    """
    model_names = list(results.keys())
    row_labels = list(class_names) + ["Overall accuracy (test)", "5-fold CV accuracy (mean ± std)"]

    columns_text = [row_labels]
    columns_fill = [["white"] * len(row_labels)]

    for name in model_names:
        per_class = results[name]["test"]["per_class_accuracy"]
        overall = results[name]["test"]["overall_accuracy"]
        cv_mean, cv_std = cv_results[name]

        text_col = []
        fill_col = []
        for cls in class_names:
            value = per_class[cls]
            text_col.append("n/a" if np.isnan(value) else f"{value:.4f}")
            fill_col.append(accuracy_to_color(value))
        text_col.append(f"{overall:.4f}")
        fill_col.append(accuracy_to_color(overall))
        text_col.append(f"{cv_mean:.4f} ± {cv_std:.4f}")
        fill_col.append(accuracy_to_color(cv_mean))

        columns_text.append(text_col)
        columns_fill.append(fill_col)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Class"] + model_names,
                    fill_color=COLOR_STATIONARY,
                    font=dict(color="white", size=13),
                    align="left",
                    height=32,
                ),
                cells=dict(
                    values=columns_text,
                    fill_color=columns_fill,
                    align="left",
                    height=26,
                    font=dict(color="black", size=12),
                ),
            )
        ]
    )
    fig.update_layout(
        title="Per-Class Test Accuracy Comparison",
        paper_bgcolor="white",
        height=60 + 10 + 32 + 26 * len(row_labels) + 20,
        width=800,
        margin=dict(t=60, b=10, l=10, r=10),
    )
    return fig


def build_html_report(fig_bar: go.Figure, fig_cm: go.Figure, fig_table: go.Figure) -> str:
    """Assemble the three Plotly figures into a single static HTML report.

    Parameters
    ----------
    fig_bar : go.Figure
        Output of ``build_accuracy_bar_figure``.
    fig_cm : go.Figure
        Output of ``build_confusion_matrix_figure``.
    fig_table : go.Figure
        Output of ``build_comparison_table_figure``.

    Returns
    -------
    str
        Complete HTML document as a string.
    """
    bar_html = fig_bar.to_html(full_html=False, include_plotlyjs="cdn", div_id="accuracy-bar")
    cm_html = fig_cm.to_html(full_html=False, include_plotlyjs=False, div_id="confusion-matrices")
    table_html = fig_table.to_html(full_html=False, include_plotlyjs=False, div_id="comparison-table")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{TITLE}</title>
<style>
  body {{
    background: #ffffff;
    color: #222222;
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 32px 48px 64px;
  }}
  h1 {{
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 4px;
  }}
  h2 {{
    font-size: 16px;
    font-weight: 600;
    color: #444444;
    margin-top: 48px;
    margin-bottom: 8px;
  }}
  .section {{
    margin-bottom: 40px;
  }}
</style>
</head>
<body>
  <h1>{TITLE}</h1>
  <div class="section">
    <h2>Per-class accuracy</h2>
    {bar_html}
  </div>
  <div class="section">
    <h2>Confusion matrices</h2>
    {cm_html}
  </div>
  <div class="section">
    <h2>Test-set comparison table</h2>
    {table_html}
  </div>
</body>
</html>
"""


def main() -> None:
    """Run the pipeline and write the interactive HTML report to disk."""
    pipeline_output = run_pipeline()
    class_names = pipeline_output["class_names"]
    results = pipeline_output["results"]
    cv_results = pipeline_output["cv_results"]

    fig_bar = build_accuracy_bar_figure(results, class_names)
    fig_cm = build_confusion_matrix_figure(results, class_names)
    fig_table = build_comparison_table_figure(results, class_names, cv_results)

    html = build_html_report(fig_bar, fig_cm, fig_table)
    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved interactive report to {OUTPUT_HTML_PATH}")


if __name__ == "__main__":
    main()
