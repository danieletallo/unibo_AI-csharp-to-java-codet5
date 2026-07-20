# CHART HELPERS
#
# Matplotlib figures for the Gradio dashboard. Same validated palette used
# throughout the project's other dashboard (dark surface, blue/aqua
# categorical pair) plus a fixed status ramp for the error-category charts.
# Figures render at dpi=150 off two shared size presets so every tab reads
# at a consistent scale when projected.

import matplotlib.pyplot as plt

SURFACE = "#1a1a19"
INK = "#ffffff"
MUTED_INK = "#c3c2b7"
GRID = "#2c2c2a"

SERIES_SMALL = "#3987e5"
# Deliberately NOT green: green is already spoken for by STATUS_GOOD (the "exact
# match" segment in the error-composition chart) and by the callout accent color
# in app.py's CSS. Using it a third time for "codet5-base" risked green quietly
# reading as "the winner" everywhere it appeared, even in charts arguing base and
# small are statistically indistinguishable. Violet is unused elsewhere.
SERIES_BASE = "#8c6fe0"
SERIES_T5VANILLA = "#c3852a"
SEQ_LIGHT = "#86b6ef"
SEQ_DARK = "#3987e5"

STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

DPI = 150
# Dedicated to plot_error_composition, which always renders one row per model x
# {fine-tuned, zero-shot}: 6 rows since t5vanilla was added, up from 4. Height is
# taller than the other wide presets so row/label spacing doesn't crowd on a
# projector.
FIGSIZE_ERROR = (12, 4.0)
FIGSIZE_TALL = (11, 3.2)
FIGSIZE_SHORT = (11, 2.6)
FIGSIZE_PAIR = (12, 3)


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED_INK)
    ax.tick_params(colors=MUTED_INK)
    ax.yaxis.label.set_color(MUTED_INK)
    ax.xaxis.label.set_color(MUTED_INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_dumbbell(families):
    """Zero-shot -> fine-tuned BLEU per model family, one dumbbell row each."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SHORT, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    labels = [f["label"] for f in families]
    for i, f in enumerate(families):
        zero_bleu = f["zero_shot"]["corpus_bleu"]
        ft_bleu = f["fine_tuned"]["corpus_bleu"]
        ax.plot([zero_bleu, ft_bleu], [i, i], color=MUTED_INK, linewidth=2, zorder=1)
        ax.scatter([zero_bleu], [i], color=SEQ_LIGHT, s=90, zorder=2, label="Zero-shot" if i == 0 else None)
        ax.scatter([ft_bleu], [i], color=SEQ_DARK, s=90, zorder=2, label="Fine-tuned" if i == 0 else None)
        ax.text(ft_bleu, i + 0.22, f"{ft_bleu:.1f}", color=INK, fontsize=9, ha="center")
        ax.text(max(zero_bleu, 6), i - 0.28, f"{zero_bleu:.2f}", color=MUTED_INK, fontsize=8.5, ha="center")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=INK)
    ax.set_ylim(-0.6, len(labels) - 1 + 0.6)
    ax.set_xlim(-3, 100)
    ax.set_xlabel("Corpus BLEU")
    ax.legend(facecolor=SURFACE, labelcolor=INK, edgecolor=MUTED_INK, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    return fig


FAMILY_COLORS = {"small": SERIES_SMALL, "base": SERIES_BASE, "t5vanilla": SERIES_T5VANILLA}


def plot_model_comparison(families):
    """Grouped bars: BLEU and Exact Match, one bar per fine-tuned model family (any count)."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PAIR, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    metrics = [("corpus_bleu", "Corpus BLEU"), ("exact_match", "Exact Match %")]
    for ax, (key, title) in zip(axes, metrics):
        values = [f["fine_tuned"][key] for f in families]
        colors = [FAMILY_COLORS.get(f["key"], MUTED_INK) for f in families]
        bars = ax.bar([f["label"] for f in families], values, color=colors, width=0.55)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", color=INK, fontsize=9)
            # A bar at (or near) zero height is easy to misread at a glance as a
            # rendering bug or missing data rather than a deliberate result (e.g.
            # t5vanilla's 0.0% Exact Match). Call it out explicitly with an arrow
            # so it reads as "measured and real", not broken.
            if key == "exact_match" and v < 1.0:
                ax.annotate(
                    "never exact",
                    xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(bar.get_x() + bar.get_width() / 2, 20),
                    ha="center",
                    fontsize=7.8,
                    color=STATUS_CRITICAL,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=STATUS_CRITICAL, linewidth=1.2),
                )
        ax.set_ylim(0, 100)
        ax.set_title(title, color=INK, fontsize=10)
        _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_decoding_comparison(beam_search):
    """Greedy -> beam search (num_beams=4) dumbbell for the chosen model, one panel each
    for BLEU and Exact Match. Each panel's x-axis is zoomed to a tight window around the
    two points (not the 0-100 range used elsewhere), because at full scale a 0.69-point
    BLEU gap and a 0.5-point EM gap would be visually indistinguishable from zero, which
    would misrepresent "small but real" as "no effect". Each title is labelled
    "(axis zoomed)" so this chart can't be mistaken for using the same 0-100 scale as the
    fine-tuning dumbbell above it in the same tab.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PAIR, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    greedy = beam_search["by_beams"][1]
    beam4 = beam_search["by_beams"][4]
    specs = [
        ("corpus_bleu", "Corpus BLEU (axis zoomed)", "{:.2f}"),
        ("exact_match", "Exact Match % (axis zoomed)", "{:.1f}"),
    ]
    legend_handles = None
    for ax, (key, title, fmt) in zip(axes, specs):
        g, b = greedy[key], beam4[key]
        pad = max((b - g) * 1.8, 0.3)
        ax.plot([g, b], [0, 0], color=MUTED_INK, linewidth=2, zorder=1)
        h_greedy = ax.scatter([g], [0], color=SEQ_LIGHT, s=140, zorder=2, label="Greedy (num_beams=1)")
        h_beam = ax.scatter([b], [0], color=SEQ_DARK, s=140, zorder=2, label="Beam search (num_beams=4)")
        ax.text(g, 0.32, fmt.format(g), color=MUTED_INK, fontsize=9.5, ha="center")
        ax.text(b, -0.42, fmt.format(b), color=INK, fontsize=9.5, ha="center", fontweight="bold")
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.set_xlim(g - pad, b + pad)
        ax.set_title(title, color=INK, fontsize=10)
        _style_axes(ax)
        if legend_handles is None:
            legend_handles = [h_greedy, h_beam]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2,
               facecolor=SURFACE, labelcolor=INK, edgecolor=MUTED_INK, fontsize=9)
    fig.tight_layout()
    return fig


def plot_error_composition(rows):
    """100%-stacked horizontal bars, one row per model, segments = outcome severity.

    rows: list of {"label": str, "good": int, "warning": int, "serious": int, "critical": int}
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_ERROR, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    labels = [r["label"] for r in rows]
    keys = [("good", STATUS_GOOD, "Exact match"), ("warning", STATUS_WARNING, "Near miss"),
            ("serious", STATUS_SERIOUS, "Other mismatch"), ("critical", STATUS_CRITICAL, "Empty output")]
    y = range(len(rows))
    left = [0] * len(rows)
    for key, color, seg_label in keys:
        totals = [r["good"] + r["warning"] + r["serious"] + r["critical"] for r in rows]
        shares = [100 * r[key] / t for r, t in zip(rows, totals)]
        ax.barh(list(y), shares, left=left, color=color, height=0.6, label=seg_label)
        left = [l + s for l, s in zip(left, shares)]
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of 1,000 test examples")
    ax.invert_yaxis()
    ax.legend(facecolor=SURFACE, labelcolor=INK, edgecolor=MUTED_INK, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8.5)
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    return fig


def plot_loss_curves(runs):
    """Eval_loss per epoch, one line per run, minimum point marked on each curve.

    runs: list of {"curve": [{"epoch":.., "eval_loss":..}, ...], "color": str, "label": str}.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    for run in runs:
        curve = run["curve"]
        color = run["color"]
        label = run["label"]
        epochs = [p["epoch"] for p in curve]
        losses = [p["eval_loss"] for p in curve]
        ax.plot(epochs, losses, color=color, linewidth=2, marker="o", markersize=4, label=label)
        min_idx = losses.index(min(losses))
        ax.scatter([epochs[min_idx]], [losses[min_idx]], color=color, s=90, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.annotate(f"{losses[min_idx]:.4f}", (epochs[min_idx], losses[min_idx]), textcoords="offset points",
                    xytext=(0, 10), ha="center", color=color, fontsize=9, fontweight="bold")
        # The min-point marker alone looks identical whether a curve converged at the
        # last epoch or simply ran out of budget while still improving; the two only
        # differ in the caption text beneath the chart. Flag the latter case directly
        # on the chart too, so the point survives even if the caption goes unread.
        if min_idx == len(losses) - 1 and len(losses) > 1:
            ax.annotate(
                "still decreasing\n(not converged)",
                (epochs[min_idx], losses[min_idx]),
                textcoords="offset points",
                xytext=(18, -16),
                ha="left",
                color=color,
                fontsize=7.8,
                fontstyle="italic",
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("eval_loss")
    ax.legend(facecolor=SURFACE, labelcolor=INK, edgecolor=MUTED_INK, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    return fig
