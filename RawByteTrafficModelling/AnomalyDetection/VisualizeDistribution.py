"""
Interactive distribution visualizer for .npy files using Plotly.

Usage:
    python visualize_distributions.py /path/to/npy/directory

Requirements:
    pip install numpy plotly scipy
"""

import sys
import os
import glob
import numpy as np
from scipy.stats import gaussian_kde, skew, kurtosis
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_npy_files(directory):
    """Load all .npy files from a directory."""
    files = sorted(glob.glob(os.path.join(directory, "*.npy")))
    if not files:
        print(f"No .npy files found in '{directory}'")
        sys.exit(1)

    data = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        arr = np.load(f).flatten().astype(float)
        # Drop NaN/Inf
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            print(f"  Skipping '{name}' (empty after cleaning)")
            continue
        data[name] = arr
        print(f"  Loaded '{name}': {len(arr)} values, "
              f"range [{arr.min():.4g}, {arr.max():.4g}]")
    return data


def compute_kde(values, n_points=512):
    """Compute a KDE curve over the data range (with 5% padding)."""
    lo, hi = values.min(), values.max()
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    x = np.linspace(lo - pad, hi + pad, n_points)
    try:
        kde = gaussian_kde(values, bw_method="scott")
        y = kde(x)
    except np.linalg.LinAlgError:
        # Fallback: all values identical or near-identical
        y = np.zeros_like(x)
    return x, y


def build_figure(data):
    """Build a multi-panel Plotly figure."""

    names = list(data.keys())
    n = len(names)

    # --- Qualitative color palette (up to 24 distinct colors) ---
    palette = (
        ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
         "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
        + [f"hsl({h}, 70%, 55%)" for h in range(0, 360, 15)]
    )

    # =====================================================================
    # Figure 1: Overlaid KDE curves (main comparison view)
    # =====================================================================
    fig_kde = go.Figure()
    for i, name in enumerate(names):
        x, y = compute_kde(data[name])
        color = palette[i % len(palette)]
        fig_kde.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=name,
            line=dict(width=2, color=color),
            hovertemplate=f"<b>{name}</b><br>x=%{{x:.4g}}<br>density=%{{y:.4g}}<extra></extra>",
        ))

    fig_kde.update_layout(
        title="Distribution Comparison — KDE Overlay",
        xaxis_title="Value", yaxis_title="Density",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        height=550,
    )

    # =====================================================================
    # Figure 2: Small-multiples (individual histograms + KDE per file)
    # =====================================================================
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig_grid = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=names,
        horizontal_spacing=0.06, vertical_spacing=0.08,
    )

    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        vals = data[name]
        color = palette[i % len(palette)]

        # Histogram
        fig_grid.add_trace(go.Histogram(
            x=vals, nbinsx=60, histnorm="probability density",
            marker_color=color, opacity=0.45,
            showlegend=False, name=name,
            hovertemplate="bin=%{x:.4g}<br>density=%{y:.4g}<extra></extra>",
        ), row=r + 1, col=c + 1)

        # KDE overlay
        x_kde, y_kde = compute_kde(vals)
        fig_grid.add_trace(go.Scatter(
            x=x_kde, y=y_kde, mode="lines",
            line=dict(color=color, width=2),
            showlegend=False, name=f"{name} KDE",
        ), row=r + 1, col=c + 1)

    fig_grid.update_layout(
        title="Individual Distributions — Histogram + KDE",
        template="plotly_white",
        height=max(350 * rows, 450),
    )

    # =====================================================================
    # Figure 3: Box / violin plot for quick shape & outlier comparison
    # =====================================================================
    fig_violin = go.Figure()
    for i, name in enumerate(names):
        color = palette[i % len(palette)]
        fig_violin.add_trace(go.Violin(
            y=data[name], name=name, box_visible=True,
            meanline_visible=True, line_color=color, opacity=0.7,
            hoverinfo="y", scalemode="width",
        ))

    fig_violin.update_layout(
        title="Violin + Box Plots",
        yaxis_title="Value",
        template="plotly_white",
        height=500,
    )

    # =====================================================================
    # Figure 4: Summary statistics table
    # =====================================================================
    stats_header = ["File", "N", "Mean", "Std", "Min", "Median", "Max",
                    "Skewness", "Kurtosis"]
    stats_rows = {h: [] for h in stats_header}
    for name in names:
        v = data[name]
        stats_rows["File"].append(name)
        stats_rows["N"].append(len(v))
        stats_rows["Mean"].append(f"{v.mean():.4g}")
        stats_rows["Std"].append(f"{v.std():.4g}")
        stats_rows["Min"].append(f"{v.min():.4g}")
        stats_rows["Median"].append(f"{np.median(v):.4g}")
        stats_rows["Max"].append(f"{v.max():.4g}")
        stats_rows["Skewness"].append(f"{skew(v):.4g}")
        stats_rows["Kurtosis"].append(f"{kurtosis(v):.4g}")

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=stats_header,
                    fill_color="paleturquoise", align="center"),
        cells=dict(values=[stats_rows[h] for h in stats_header],
                   fill_color="lavender", align="center"),
    )])
    fig_table.update_layout(title="Summary Statistics", height=max(200 + 30 * n, 350))

    return fig_kde, fig_grid, fig_violin, fig_table


def main(directory):
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        sys.exit(1)

    print(f"Scanning '{directory}' for .npy files …")
    data = load_npy_files(directory)
    if not data:
        print("No valid data loaded.")
        sys.exit(1)

    print(f"\nBuilding figures for {len(data)} file(s) …")
    fig_kde, fig_grid, fig_violin, fig_table = build_figure(data)

    out = os.path.join(directory, "distributions.html")
    with open(out, "w") as f:
        f.write("<html><head><meta charset='utf-8'>"
                "<title>Distribution Report</title></head><body>\n")
        f.write(fig_kde.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("<hr>\n")
        f.write(fig_grid.to_html(full_html=False, include_plotlyjs=False))
        f.write("<hr>\n")
        f.write(fig_violin.to_html(full_html=False, include_plotlyjs=False))
        f.write("<hr>\n")
        f.write(fig_table.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>\n")

    print(f"\nDone! Open the report:\n  {os.path.abspath(out)}")


if __name__ == "__main__":
    directory = "RawByteTrafficModelling/AnomalyDetection/Outputs"
    main(directory)