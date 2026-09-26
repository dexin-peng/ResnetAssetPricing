"""Figures rendered only from completed analysis caches."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.transforms import ScaledTranslation
from utils.paper_source.artifact_cache import load_render_cache as load_cache
from .refinement_cache import load_depth_summary
from .style import MODEL_COLORS, init_mpl, savefig_pair


def generate_rolling_train_test_split(*, usetex: bool = False) -> None:
    init_mpl(usetex=False)
    payload = load_cache('RollingTrainTestSplit')
    if payload is None:
        return
    train_x = payload['train_x']
    train_y = payload['train_y']
    test_x = payload['test_x']
    test_y = payload['test_y']
    fig, ax = plt.subplots(figsize=(6.9, 2.9))
    ax.scatter(train_x, train_y, s=4.2, color="#1f77b4", label="Training Sample", linewidths=0)
    ax.scatter(test_x, test_y, s=5.2, color="#ff7f0e", label="Test Sample", linewidths=0)
    ax.set_xlim(1960, 2026)
    ax.set_ylim(1985, 2026)
    ax.set_xlabel("Sample Year")
    ax.set_ylabel("Forecast Origin Year")
    ax.set_xticks([1963, 1973, 1983, 1993, 2003, 2013, 2023])
    ax.set_yticks([1987, 1993, 1999, 2005, 2011, 2017, 2023])
    ax.grid(True, linestyle="--", color="#bdbdbd", linewidth=0.55)
    ax.legend(loc="lower right", frameon=True, markerscale=1.2)
    ax.annotate(
        "",
        xy=(0.0, 1.02),
        xytext=(0.0, 1.0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, mutation_scale=18),
        annotation_clip=False,
    )
    fig.tight_layout()
    savefig_pair("RollingTrainTestSplit", fig)
    plt.close(fig)


def _forecast_refinement_depth_interval_frame(metric: str, cache_dir=None) -> pd.DataFrame:
    """Select display-ready estimates; all aggregation and inference is upstream."""
    source = load_depth_summary(cache_dir)
    frame = source[source["Metric"].eq(metric)].copy()
    if len(frame) != 60 or frame.duplicated(["Model", "Depth"]).any():
        raise ValueError(f"Incomplete cached depth grid for {metric}.")
    if not np.isfinite(frame["Estimate"].to_numpy(dtype=float)).all():
        raise ValueError(f"At least one cached {metric} depth estimate is unavailable; it cannot be plotted as zero.")
    return frame.sort_values(["Model", "Depth"]).reset_index(drop=True)


def generate_forecast_rank_preservation_and_refinement_by_depth(
    *,
    usetex: bool = False,
    cache_dir=None,
) -> None:
    """Plot cached pairwise preservation and refinement by depth."""
    init_mpl(usetex=usetex)
    preservation = _forecast_refinement_depth_interval_frame("Preservation", cache_dir)
    refinement = _forecast_refinement_depth_interval_frame("Refinement", cache_dir)
    panels = (
        (
            "A. ResNet vs NN",
            ("ResNet", "NN"),
            (-7.0, 7.0),
            0.04,
            0.84,
            0.36,
            "lower right",
            0.03,
        ),
        (
            "B. ResNet+ vs NN+",
            ("ResNet+", "NN+"),
            (-7.0, 7.0),
            0.04,
            0.96,
            0.36,
            "lower right",
            0.03,
        ),
    )
    depths = np.arange(6, 21, dtype=float)
    fig, preservation_axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4),
        sharex=False,
        sharey=False,
        gridspec_kw={"wspace": 0.35},
    )
    refinement_axes = [preservation_ax.twinx() for preservation_ax in preservation_axes]
    preservation_handle = plt.Line2D(
        [0, 1],
        [0, 0],
        color="#333333",
        linestyle="-",
        linewidth=1.75,
        label="Preservation score",
    )
    refinement_handle = plt.Line2D(
        [0, 1],
        [0, 0],
        color="#333333",
        linestyle=(0, (3.2, 1.7)),
        linewidth=1.45,
        label="Revision accuracy",
    )
    for (
        title,
        models,
        refinement_limits,
        depth_label_y,
        preservation_legend_y,
        refinement_legend_y,
        model_legend_location,
        model_legend_y,
    ), preservation_ax, refinement_ax in zip(
        panels, preservation_axes, refinement_axes
    ):
        refinement_ax.patch.set_visible(False)
        metric_specs = (
            ("Preservation", preservation, preservation_ax, "-", 1.75, True),
            ("Refinement", refinement, refinement_ax, (0, (3.2, 1.7)), 1.45, False),
        )
        for metric, frame, ax, linestyle, linewidth, filled_marker in metric_specs:
            for model in models:
                subset = frame[frame["Model"].eq(model)].sort_values("Depth")
                if len(subset) != 15:
                    raise ValueError(f"Expected 15 depth points for {model} {metric.lower()}.")
                depth = subset["Depth"].to_numpy(dtype=float)
                if not np.array_equal(depth, depths):
                    raise ValueError(f"Unexpected depth grid for {model} {metric.lower()}.")
                estimate = subset["Estimate"].to_numpy(dtype=float)
                plotted_estimate = estimate
                overflow_high = np.zeros(len(estimate), dtype=bool)
                overflow_low = np.zeros(len(estimate), dtype=bool)
                if metric == "Refinement":
                    overflow_high = estimate > refinement_limits[1]
                    overflow_low = estimate < refinement_limits[0]
                    plotted_estimate = np.clip(
                        estimate,
                        refinement_limits[0] + 0.005 * (refinement_limits[1] - refinement_limits[0]),
                        refinement_limits[1] - 0.005 * (refinement_limits[1] - refinement_limits[0]),
                    )
                color = MODEL_COLORS[model]
                ax.plot(
                    depth,
                    plotted_estimate,
                    color=color,
                    marker="o",
                    markerfacecolor=color if filled_marker else "white",
                    markeredgecolor=color,
                    markeredgewidth=0.75,
                    markersize=4.0,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=f"{metric}: {model}",
                    zorder=4 if filled_marker else 3,
                )
                if overflow_high.any():
                    ax.scatter(
                        depth[overflow_high],
                        plotted_estimate[overflow_high],
                        color=color,
                        marker="^",
                        s=25.0,
                        linewidths=0.0,
                        zorder=6,
                    )
                if overflow_low.any():
                    ax.scatter(
                        depth[overflow_low],
                        plotted_estimate[overflow_low],
                        color=color,
                        marker="v",
                        s=25.0,
                        linewidths=0.0,
                        zorder=6,
                    )

        preservation_ax.axhline(0.0, color="#333333", linewidth=0.75, alpha=0.75, zorder=1)
        refinement_ax.axhline(
            0.0,
            color="#555555",
            linestyle="-",
            linewidth=0.95,
            alpha=0.9,
            zorder=2,
        )
        preservation_ax.axvline(
            10.5,
            color="#555555",
            linestyle=(0, (4.0, 2.0)),
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )
        for x_position, label, horizontal_alignment in (
            (10.30, "Medium", "right"),
            (10.70, "Deep", "left"),
        ):
            preservation_ax.text(
                x_position,
                depth_label_y,
                label,
                transform=preservation_ax.get_xaxis_transform(),
                color="#555555",
                fontsize=7.4,
                fontweight="bold",
                ha=horizontal_alignment,
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.5},
                zorder=5,
            )

        preservation_ax.set_title(title, fontsize=9.5, pad=15.0)
        preservation_ax.set_ylabel(r"Preservation score ($P_t$)")
        refinement_label_x = 1.10
        refinement_ax.set_ylabel("")
        for label, label_y in (
            ("Refinement", 0.80),
            ("Deterioration", 0.20),
        ):
            refinement_ax.text(
                refinement_label_x,
                label_y,
                label,
                transform=refinement_ax.transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=refinement_ax.yaxis.label.get_fontsize(),
                clip_on=False,
            )
        preservation_ax.set_xlim(5.55, 20.45)
        preservation_ax.set_ylim(0.0, 1.0)
        refinement_ax.set_ylim(*refinement_limits)
        preservation_ax.set_xticks(np.arange(6.0, 21.0, 2.0))
        preservation_ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        refinement_ax.set_yticks(np.arange(-6.0, 7.0, 3.0))
        preservation_ax.grid(False, axis="x")
        preservation_ax.grid(True, axis="y", color="#ececec", linestyle="-", linewidth=0.55)
        refinement_ax.grid(False)
        preservation_ax.set_axisbelow(True)
        refinement_ax.spines["right"].set_visible(True)
        refinement_ax.spines["right"].set_color("#333333")
        refinement_ax.spines["right"].set_linewidth(0.65)
        for arrow_start_y, arrow_end_y in ((0.985, 1.018), (0.0, -0.035)):
            refinement_ax.annotate(
                "",
                xy=(1.0, arrow_end_y),
                xytext=(1.0, arrow_start_y),
                xycoords="axes fraction",
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#333333",
                    "linewidth": 0.65,
                    "mutation_scale": 7.0,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                annotation_clip=False,
                zorder=5,
            )
        refinement_ax.tick_params(
            axis="y",
            which="major",
            right=True,
            labelright=True,
            direction="out",
            length=3.2,
            width=0.65,
            pad=2.5,
        )
        negative_tick_shift = ScaledTranslation(
            -1.5 / 72.0,
            -0.8 / 72.0,
            fig.dpi_scale_trans,
        )
        for tick_value, tick_label in zip(
            refinement_ax.get_yticks(), refinement_ax.get_yticklabels()
        ):
            if np.isclose(tick_value, -3.0) or np.isclose(tick_value, -6.0):
                tick_label.set_transform(tick_label.get_transform() + negative_tick_shift)
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                color=MODEL_COLORS[model],
                marker="o",
                markerfacecolor=MODEL_COLORS[model],
                markeredgecolor=MODEL_COLORS[model],
                markeredgewidth=0.9,
                markersize=5.0,
                linestyle="none",
                label=model,
            )
            for model in models
        ]
        model_legend = preservation_ax.legend(
            handles=legend_handles,
            frameon=False,
            loc=model_legend_location,
            bbox_to_anchor=(0.98, model_legend_y),
            ncol=2,
            prop={"size": 7.6, "weight": "bold"},
            labelcolor="linecolor",
            handlelength=1.0,
            handletextpad=0.45,
            columnspacing=1.1,
            labelspacing=0.35,
        )
        preservation_ax.add_artist(model_legend)
        preservation_legend = preservation_ax.legend(
            handles=[preservation_handle],
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.03, preservation_legend_y),
            borderaxespad=0.0,
            fontsize=7.0,
            handlelength=2.0,
            handletextpad=0.55,
        )
        preservation_legend.set_zorder(10)
        refinement_legend = refinement_ax.legend(
            handles=[refinement_handle],
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.03, refinement_legend_y),
            borderaxespad=0.0,
            fontsize=7.0,
            handlelength=2.0,
            handletextpad=0.55,
        )
        refinement_legend.set_zorder(10)
        if refinement[
            refinement["Model"].isin(models)
        ]["Estimate"].gt(refinement_limits[1]).any():
            refinement_ax.text(
                0.98,
                0.03,
                rf"$\triangle$: above $+{refinement_limits[1]:g}$ score points",
                transform=refinement_ax.transAxes,
                color="#555555",
                fontsize=7.0,
                ha="right",
                va="bottom",
            )

    for preservation_ax in preservation_axes:
        preservation_ax.set_xlabel("Depth")
    fig.subplots_adjust(wspace=0.35)
    savefig_pair("ForecastRankPreservationAndRefinementByDepth", fig)
    plt.close(fig)


def _parameter_label(value: float) -> str:
    if value >= 1_000_000_000:
        scaled = value / 1_000_000_000
        return f"{scaled:.1f}B" if scaled < 10 else f"{scaled:.0f}B"
    if value >= 1_000_000:
        scaled = value / 1_000_000
        return f"{scaled:.1f}M" if scaled < 10 else f"{scaled:.0f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def generate_parameter_scale_gap(*, usetex: bool = False) -> None:
    payload = load_cache("ParameterScaleGap")
    if payload is None:
        return
    records = payload["records"]
    if records.empty:
        return
    init_mpl(usetex=usetex)
    colors = {"Asset Pricing": "#1f7a5c", "External ML": "#777777"}
    y_positions = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(8.9, 6.1))
    ax.axvspan(0.8, 4_000_000, color="#1f7a5c", alpha=0.07, zorder=0)
    ax.axvspan(30_000_000, 700_000_000_000, color="#777777", alpha=0.045, zorder=0)
    for y, row in zip(y_positions, records.itertuples(index=False)):
        color = colors.get(row.domain, "#444444")
        low = float(row.low)
        high = float(row.high)
        ax.plot([low, high], [y, y], color=color, linewidth=2.0, solid_capstyle="round")
        ax.scatter([low, high], [y, y], color=color, s=22, zorder=3, edgecolor="white", linewidth=0.6)
        text = _parameter_label(high) if low == high else f"{_parameter_label(low)}-{_parameter_label(high)}"
        ax.text(high * 1.12, y - 0.19, text, va="top", ha="left", fontsize=8.0, color=color)
        year = str(getattr(row, "year", "") or "").strip()
        if year:
            year_x = np.sqrt(low * high)
            if str(row.label) == "CAPM":
                year_x *= 1.35
            ax.text(
                year_x,
                y + 0.21,
                year,
                va="bottom",
                ha="center",
                fontsize=7.2,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.3},
            )
    ax.set_xscale("log")
    ax.set_xlim(0.7, 900_000_000_000)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=13))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=120))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="major", length=4.0, width=0.6, color="#555555")
    ax.tick_params(axis="x", which="minor", length=2.2, width=0.45, color="#888888")
    ax.set_yticks(y_positions)
    row_labels = [
        f"{row.label} ({'regression parameters' if row.label in ('CAPM', 'Fama-French three factor', 'Carhart four factor') else 'factors' if row.label == 'Barra USE4' else 'predictors' if row.label == 'Factor zoo' else 'parameters'})"
        for row in records.itertuples(index=False)
    ]
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Count (log scale)")
    ax.set_axisbelow(True)
    ax.grid(False, axis="x")
    ax.grid(True, which="major", axis="y", color="#d0d0d0", linestyle=(0, (3.0, 3.0)), linewidth=0.55, alpha=0.85)
    for tick in ax.get_yticklabels():
        label = tick.get_text()
        if "grid" in label:
            tick.set_color("#1f7a5c")
            tick.set_fontweight("bold")
    handles = [
        plt.Line2D([0], [0], color=colors["Asset Pricing"], linewidth=2.0, label="Asset Pricing"),
        plt.Line2D([0], [0], color=colors["External ML"], linewidth=2.0, label="External ML"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    savefig_pair("ParameterScaleGap", fig)
    plt.close(fig)


def _plot_decile_return_risk_summary(
    *,
    values_by_family: dict,
    fig_name: str,
) -> None:
    fig = plt.figure(figsize=(10, 4.0))
    outer_grid = fig.add_gridspec(1, 2, wspace=0.35)
    comparison_specs = (
        ("A. ResNet vs NN", "ResNet", "NN"),
        ("B. ResNet+ vs NN+", "ResNet+", "NN+"),
    )
    decile_positions = np.arange(1, 11, dtype=float)
    bar_width = 0.42
    for col_idx, (comparison, left_family, right_family) in enumerate(comparison_specs):
        panel_grid = outer_grid[0, col_idx].subgridspec(
            2,
            1,
            height_ratios=(2.15, 1.0),
            hspace=0.06,
        )
        return_ax = fig.add_subplot(panel_grid[0, 0])
        volatility_ax = return_ax.twinx()
        sharpe_ax = fig.add_subplot(panel_grid[1, 0], sharex=return_ax)

        for family in (left_family, right_family):
            returns = values_by_family[family]["Mean"]
            volatility = values_by_family[family]["Volatility"]
            if not returns.empty:
                return_ax.plot(
                    returns.index.to_numpy(dtype=float),
                    returns.to_numpy(dtype=float),
                    marker="o",
                    linewidth=1.45,
                    markersize=3.4,
                    color=MODEL_COLORS[family],
                    linestyle="-",
                    label=f"{family} return",
                    zorder=4,
                )
            if not volatility.empty:
                volatility_ax.plot(
                    volatility.index.to_numpy(dtype=float),
                    volatility.to_numpy(dtype=float),
                    marker="o",
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    linewidth=1.15,
                    markersize=3.2,
                    color=MODEL_COLORS[family],
                    linestyle=(0, (3.2, 1.7)),
                    label=f"{family} vol.",
                    zorder=3,
                )

        for family_idx, family in enumerate((left_family, right_family)):
            sharpe = values_by_family[family]["SR"]
            if sharpe.empty:
                continue
            centers = decile_positions + (family_idx - 0.5) * bar_width
            sharpe_ax.bar(
                centers,
                sharpe.to_numpy(dtype=float),
                width=bar_width,
                align="center",
                color=to_rgba(MODEL_COLORS[family], 0.7),
                edgecolor=MODEL_COLORS[family],
                linewidth=0.65,
                hatch=None,
                label=family,
                zorder=3,
            )

        return_ax.axhline(0.0, color="black", linewidth=0.65, alpha=0.5, zorder=1)
        sharpe_ax.axhline(0.0, color="black", linewidth=0.65, alpha=0.5, zorder=2)
        return_ax.set_title(comparison)
        return_ax.set_ylabel("Mean return (\\%)")
        volatility_ax.set_ylabel("Volatility (\\%)", labelpad=7.5)
        sharpe_ax.set_ylabel("SR")
        sharpe_ax.set_xlabel("Forecast Decile")
        return_ax.set_ylim(-1.6, 2.3)
        volatility_ax.set_ylim(0.0, 9.5)
        volatility_ax.set_yticks(np.arange(0.0, 10.0, 2.0))
        sharpe_ax.set_ylim(-0.65, 0.95)
        sharpe_ax.set_xticks(range(1, 11))
        sharpe_ax.set_xlim(0.55, 10.45)
        return_ax.tick_params(axis="x", which="both", labelbottom=False)
        volatility_ax.tick_params(
            axis="y",
            which="major",
            right=True,
            labelright=True,
            direction="out",
            length=3.2,
            width=0.65,
            pad=2.5,
        )
        volatility_ax.spines["right"].set_visible(True)
        volatility_ax.spines["right"].set_color("#333333")
        volatility_ax.spines["right"].set_linewidth(0.65)
        return_ax.grid(False)
        volatility_ax.grid(False)
        sharpe_ax.grid(False)
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                color=MODEL_COLORS[family],
                marker="o",
                markerfacecolor=MODEL_COLORS[family],
                markeredgecolor=MODEL_COLORS[family],
                markeredgewidth=0.9,
                markersize=5.0,
                linestyle="none",
                label=family,
            )
            for family in (left_family, right_family)
        ]
        return_handle = plt.Line2D(
            [0, 1],
            [0, 0],
            color="#333333",
            linestyle="-",
            linewidth=1.45,
            label="Mean return (left)",
        )
        volatility_handle = plt.Line2D(
            [0, 1],
            [0, 0],
            color="#333333",
            linestyle=(0, (3.2, 1.7)),
            linewidth=1.15,
            label="Vol. (right)",
        )
        model_legend = return_ax.legend(
            handles=legend_handles,
            frameon=False,
            ncol=2,
            loc="lower right",
            prop={"size": 7.6, "weight": "bold"},
            labelcolor="linecolor",
            handlelength=1.0,
            handletextpad=0.45,
            columnspacing=1.1,
            labelspacing=0.35,
        )
        return_ax.add_artist(model_legend)
        return_legend = return_ax.legend(
            handles=[return_handle],
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.15, 0.2),
            borderaxespad=0.0,
            fontsize=7.0,
            handlelength=2.0,
            handletextpad=0.55,
        )
        return_ax.add_artist(return_legend)
        return_ax.legend(
            handles=[volatility_handle],
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.15, 0.85),
            borderaxespad=0.0,
            fontsize=7.0,
            handlelength=2.0,
            handletextpad=0.55,
        )

    savefig_pair(fig_name, fig)
    plt.close(fig)


def generate_decile_monotonicity_selected(*, usetex: bool = False) -> None:
    init_mpl(usetex=usetex)
    payload = load_cache("DecileMonotonicitySelected")
    _plot_decile_return_risk_summary(values_by_family=payload["values"], fig_name="DecileMonotonicitySelected")


def _draw_fallback_crosses(
    ax: plt.Axes,
    cells: list[tuple[int, int]],
    *,
    depth_positions: dict[int, int],
    seed_positions: dict[int, int],
) -> None:
    dash = (0, (2.2, 1.2))
    for seed, depth in cells:
        if seed not in seed_positions or depth not in depth_positions:
            continue
        x = depth_positions[depth]
        y = seed_positions[seed]
        ax.plot(
            [x - 0.27, x + 0.27],
            [y, y],
            color="#007a3d",
            linewidth=0.85,
            linestyle=dash,
            alpha=0.4,
            zorder=8,
            solid_capstyle="round",
        )
        ax.plot(
            [x, x],
            [y - 0.27, y + 0.27],
            color="#007a3d",
            linewidth=0.85,
            linestyle=dash,
            alpha=0.4,
            zorder=8,
            solid_capstyle="round",
        )


def _delta_heatmap_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "resnetp_delta",
        [to_rgba("#ad5759", alpha=0.68), "#f7f7f2", "#1f7a5c"],
    )
    cmap.set_bad("#eeeeee")
    return cmap


def _format_delta_tick(value: float, _pos: int) -> str:
    if abs(value) < 5e-3:
        return "0"
    sign = "+" if value > 0 else "-"
    if abs(value) < 1:
        return f"{sign}{abs(value):.1f}"
    return f"{sign}{abs(value):.0f}"


def _draw_delta_heatmap_axis(
    ax: plt.Axes,
    *,
    matrix: pd.DataFrame,
    depths: list[int],
    seeds: list[int],
    left_family: str,
    right_family: str,
    metric: str,
    title: str | None,
    vmax: float,
    fallback_cells: list[tuple[int, int]],
) -> object:
    data = matrix.to_numpy(dtype=float)
    mask = np.isnan(data)
    im = ax.imshow(
        np.ma.array(data, mask=mask),
        aspect="auto",
        cmap=_delta_heatmap_cmap(),
        vmin=-vmax,
        vmax=vmax,
    )
    if title:
        ax.set_title(title, fontsize=11.8, pad=22)
    ax.set_xlabel("Depth", fontsize=11.2)
    ax.set_ylabel("Width Schedule", fontsize=11.2)
    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_yticks(np.arange(len(seeds)))
    ax.set_yticklabels([str(s) for s in seeds])
    ax.tick_params(axis="both", labelsize=9.8)
    xlabels = ax.get_xticklabels()
    for j, _depth in enumerate(depths):
        if j % 2 == 1:
            xlabels[j].set_visible(False)
    depth_positions = {depth: pos for pos, depth in enumerate(depths)}
    seed_positions = {seed: pos for pos, seed in enumerate(seeds)}
    _draw_fallback_crosses(
        ax,
        fallback_cells,
        depth_positions=depth_positions,
        seed_positions=seed_positions,
    )
    diagonal_support = sorted(set(depth_positions).intersection(seed_positions))
    if diagonal_support:
        start = diagonal_support[0]
        end = diagonal_support[-1]
        ax.plot(
            [depth_positions[start] - 0.5, depth_positions[end] + 0.5],
            [seed_positions[start] - 0.5, seed_positions[end] + 0.5],
            color="#c7c7c7",
            linewidth=1.35,
            alpha=0.95,
            solid_capstyle="round",
            zorder=5,
        )
    for boundary in (5, 10):
        if boundary in depth_positions and boundary + 1 in depth_positions:
            ax.axvline(depth_positions[boundary] + 0.5, color="white", linewidth=1.1, alpha=0.9)
    bands = [("Shallow", 1, 5), ("Medium", 6, 10), ("Deep", 11, 20)]
    for label, lo, hi in bands:
        band_depths = [depth for depth in depths if lo <= depth <= hi]
        if not band_depths:
            continue
        center = np.mean([depth_positions[depth] for depth in band_depths])
        ax.text(
            center,
            1.012,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9.6,
        )
    return im


def generate_specification_delta_sr_heatmaps(*, usetex: bool = False) -> None:
    init_mpl(usetex=usetex)
    metric = "Value Weighted Long Short Sharpe Ratio"
    displays = [
        {
            "left_family": "ResNet",
            "right_family": "NN",
            "title": "A. ResNet SR minus NN SR",
        },
        {
            "left_family": "ResNet+",
            "right_family": "NN+",
            "title": "B. ResNet+ SR minus NN+ SR",
        },
    ]
    cached_panels = load_cache("SpecificationDeltaSRHeatmaps")["panels"]
    payloads = []
    for display in displays:
        cached = cached_panels[display["left_family"], display["right_family"]]
        matrix, depths, seeds, max_abs = (cached[key] for key in ("matrix", "depths", "seeds", "max_abs"))
        if matrix.empty:
            return
        vmax = max(float(max_abs), 0.05) if np.isfinite(max_abs) else 0.05
        payloads.append((display, matrix, depths, seeds, vmax))

    n_panels = len(payloads)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9.2, 2.4 * n_panels))
    axes = np.atleast_1d(axes)
    images = []
    for panel_idx, (ax, (display, matrix, depths, seeds, vmax)) in enumerate(zip(axes, payloads)):
        image = _draw_delta_heatmap_axis(
            ax,
            matrix=matrix,
            depths=depths,
            seeds=seeds,
            left_family=str(display["left_family"]),
            right_family=str(display["right_family"]),
            metric=metric,
            title=str(display["title"]),
            vmax=vmax,
            fallback_cells=cached_panels[display["left_family"], display["right_family"]]["fallback_cells"],
        )
        if panel_idx < n_panels - 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", labelbottom=False)
        images.append((ax, image))
    fig.subplots_adjust(left=0.075, right=0.89, top=0.955, bottom=0.055, hspace=0.42)
    for ax, image in images:
        pos = ax.get_position()
        cbar_ax = fig.add_axes([0.91, pos.y0, 0.022, pos.height])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label("$\\Delta$ SR", fontsize=11.2)
        cbar.formatter = FuncFormatter(_format_delta_tick)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=9.8, length=0, pad=5)
    savefig_pair("SpecificationDeltaSRHeatmaps", fig)
    plt.close(fig)


def generate_all_figures(*, usetex: bool = False) -> None:
    from utils.paper_source.artifact_registry import ARTIFACTS
    import importlib
    for asset, record in ARTIFACTS.items():
        if record[4] == "figure":
            print(f"[render] {asset}", flush=True)
            getattr(importlib.import_module(record[2]), record[3])(usetex=usetex)
