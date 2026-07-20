# GRADIO DASHBOARD
#
# Presents Phase A-D of the C# -> Java project for a live presentation:
# dataset facts, training setup, Phase C evaluation (fine-tuning impact,
# small vs base, decoding strategy, error composition), and Phase D's
# ablation study (128 vs 256 tokens). Reads the same JSON files Phase C/D
# already save, nothing here is recomputed.
#
# Styling is forced to a single dark look via custom CSS instead of relying
# on Gradio's automatic light/dark switching, since the charts hardcode a
# dark surface and a live demo shouldn't depend on the presenter's laptop
# being in dark mode.

import random

import gradio as gr

from data import (
    PHASE_A, PHASE_B, load_phase_c, load_phase_d, load_beam_search, load_beam_search_predictions,
    load_phase_b_training, load_pretraining_loss_curves,
)
from plots import (
    plot_decoding_comparison, plot_dumbbell, plot_error_composition, plot_loss_curves,
    plot_model_comparison, SERIES_SMALL, SERIES_BASE, SERIES_T5VANILLA, SEQ_DARK, SEQ_LIGHT,
)

phase_c = load_phase_c()
phase_d = load_phase_d()
beam_search = load_beam_search()
beam_predictions = load_beam_search_predictions()
example_pool = beam_predictions["examples"]  # 1,000 rows, unfiltered: random.choice draws from all of them
phase_b_training = load_phase_b_training()
pretraining_loss_curves = load_pretraining_loss_curves()

# metrics.json's own best_finetuned_model now picks codet5-base by raw corpus BLEU
# alone (89.49, the highest of the three fine-tuned models), so no manual override
# is needed here anymore. Its Exact Match (64.2%) is a hair below codet5-small's
# (64.8%): see report.txt for why that razor-thin gap (0.6pp, 6 examples out of
# 1,000) isn't treated as a real difference.
BEST_FINETUNED_LABEL = "codet5-base"
best_family = next(f for f in phase_c["families"] if f["label"] == BEST_FINETUNED_LABEL)
t5vanilla_family = next(f for f in phase_c["families"] if f["key"] == "t5vanilla")

# Code-pretraining effect: same size as codet5-base (t5-base, ~220M params),
# same fine-tuning recipe, the only variable that changes is whether the
# pretraining corpus was code (CodeT5) or generic text (vanilla T5).
pretrain_bleu_delta = best_family["fine_tuned"]["corpus_bleu"] - t5vanilla_family["fine_tuned"]["corpus_bleu"]
pretrain_em_delta = best_family["fine_tuned"]["exact_match"] - t5vanilla_family["fine_tuned"]["exact_match"]
bleu_delta = phase_d["len128"]["corpus_bleu"] - phase_d["len256"]["corpus_bleu"]
em_delta = phase_d["len128"]["exact_match"] - phase_d["len256"]["exact_match"]
time_delta_pct = 100 * (phase_d["len128"]["train_runtime_seconds"] - phase_d["len256"]["train_runtime_seconds"]) / phase_d["len256"]["train_runtime_seconds"]
flos_delta_pct = 100 * (phase_d["len128"]["total_flos"] - phase_d["len256"]["total_flos"]) / phase_d["len256"]["total_flos"]
train_minutes_256 = phase_d["len256"]["train_runtime_seconds"] / 60
train_minutes_128 = phase_d["len128"]["train_runtime_seconds"] / 60

# t5vanilla vs codet5-base training compute, expressed as a percentage (mirrors
# flos_delta_pct above for the ablation tab) rather than leaving raw scientific
# notation as the only figure shown on screen.
flos_pretrain_delta_pct = 100 * (
    phase_b_training["t5vanilla"]["total_flos"] - phase_b_training["base"]["total_flos"]
) / phase_b_training["base"]["total_flos"]

# Beam search (num_beams=4) vs greedy (num_beams=1), same chosen fine-tuned model.
# A small, direction-confirmed gain at a measured (not estimated) inference cost:
# see the Results tab callout below for the full hedge (no significance test, tiny
# vs fine-tuning/ablation).
beam_greedy = beam_search["by_beams"][1]
beam_beam4 = beam_search["by_beams"][4]
beam_bleu_delta = beam_beam4["corpus_bleu"] - beam_greedy["corpus_bleu"]
beam_em_delta = beam_beam4["exact_match"] - beam_greedy["exact_match"]
beam_bleu_delta_rel_pct = 100 * beam_bleu_delta / beam_greedy["corpus_bleu"]
beam_time_delta_pct = 100 * (beam_beam4["elapsed_seconds"] - beam_greedy["elapsed_seconds"]) / beam_greedy["elapsed_seconds"]

error_rows = [
    {"label": "small - fine-tuned", **{
        "good": phase_c["by_name"]["codet5-small (fine-tuned)"]["exact_match"] * 10,
        "warning": phase_c["by_name"]["codet5-small (fine-tuned)"]["error_categories"]["near_miss"],
        "serious": phase_c["by_name"]["codet5-small (fine-tuned)"]["error_categories"]["other_mismatch"],
        "critical": phase_c["by_name"]["codet5-small (fine-tuned)"]["error_categories"]["empty_output"],
    }},
    {"label": "base - fine-tuned", **{
        "good": phase_c["by_name"]["codet5-base (fine-tuned)"]["exact_match"] * 10,
        "warning": phase_c["by_name"]["codet5-base (fine-tuned)"]["error_categories"]["near_miss"],
        "serious": phase_c["by_name"]["codet5-base (fine-tuned)"]["error_categories"]["other_mismatch"],
        "critical": phase_c["by_name"]["codet5-base (fine-tuned)"]["error_categories"]["empty_output"],
    }},
    {"label": "small - zero-shot", **{
        "good": phase_c["by_name"]["codet5-small (zero-shot)"]["exact_match"] * 10,
        "warning": phase_c["by_name"]["codet5-small (zero-shot)"]["error_categories"]["near_miss"],
        "serious": phase_c["by_name"]["codet5-small (zero-shot)"]["error_categories"]["other_mismatch"],
        "critical": phase_c["by_name"]["codet5-small (zero-shot)"]["error_categories"]["empty_output"],
    }},
    {"label": "base - zero-shot", **{
        "good": phase_c["by_name"]["codet5-base (zero-shot)"]["exact_match"] * 10,
        "warning": phase_c["by_name"]["codet5-base (zero-shot)"]["error_categories"]["near_miss"],
        "serious": phase_c["by_name"]["codet5-base (zero-shot)"]["error_categories"]["other_mismatch"],
        "critical": phase_c["by_name"]["codet5-base (zero-shot)"]["error_categories"]["empty_output"],
    }},
    {"label": "t5vanilla - fine-tuned", **{
        "good": phase_c["by_name"]["t5vanilla (fine-tuned)"]["exact_match"] * 10,
        "warning": phase_c["by_name"]["t5vanilla (fine-tuned)"]["error_categories"]["near_miss"],
        "serious": phase_c["by_name"]["t5vanilla (fine-tuned)"]["error_categories"]["other_mismatch"],
        "critical": phase_c["by_name"]["t5vanilla (fine-tuned)"]["error_categories"]["empty_output"],
    }},
    {"label": "t5vanilla - zero-shot", **{
        "good": phase_c["by_name"]["t5vanilla (zero-shot)"]["exact_match"] * 10,
        "warning": phase_c["by_name"]["t5vanilla (zero-shot)"]["error_categories"]["near_miss"],
        "serious": phase_c["by_name"]["t5vanilla (zero-shot)"]["error_categories"]["other_mismatch"],
        "critical": phase_c["by_name"]["t5vanilla (zero-shot)"]["error_categories"]["empty_output"],
    }},
]


def stat_card(value, label, delta=None, tone="neutral"):
    """One KPI tile: a headline value, a label underneath, an optional colored delta chip."""
    chip = f'<div class="stat-delta">{delta}</div>' if delta else ""
    return (
        f'<div class="stat-card stat-tone-{tone}">'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f"{chip}"
        f"</div>"
    )


def stat_row(*cards):
    return '<div class="stat-row">' + "".join(cards) + "</div>"


def ablation_card(value, label, good):
    """Delta tile for the ablation tab: 'good' means 128 tokens wins on this metric."""
    tone = "good" if good else "bad"
    chip = "Better at 128" if good else "Worse at 128"
    return stat_card(value, label, delta=chip, tone=tone)


def note_card(title, body):
    return f'<div class="note-card"><div class="note-title">{title}</div><div class="note-body">{body}</div></div>'


def pick_random_example():
    """Uniformly random pick from all 1,000 precomputed beam-search (num_beams=4) rows:
    no exact_match filter, no curated subset, no reroll-until-a-good-one logic within a
    single click. Honesty feature, not a highlight reel: see beam_search_predictions.json
    and the representativeness caveat in the Results / Example Browser copy.

    NOTE: this function's own fairness only covers what happens inside one click. It
    cannot stop a presenter from clicking several times off-screen before going live and
    only pressing the button in front of the audience once a favorable (exact-match) row
    shows up: that guarantee lives in presenter discipline (see the "Cosa NON fare" note
    in discorso_presentazione.md), not in this code.

    Returns a 5-tuple matching the Example Browser tab's five output components:
    (cs_code, java_ref_code, java_pred_code, stats_html, aggregate_html).
    """
    ex = random.choice(example_pool)
    is_match = ex["exact_match"]
    # Tone: "good" for an exact match, but deliberately NOT "bad" for a miss; misses are
    # ~1/3 of the test set by design (see the tab's strapline), not failures, so branding
    # them red would fight the tab's own "honesty feature" framing. Reuses the existing
    # stat-tone-neutral CSS class rather than introducing a new color.
    tone = "good" if is_match else "neutral"
    match_label = "Yes" if is_match else "No"

    stats_html = stat_row(
        stat_card(f"#{ex['index']}", "Test example (of 1,000)"),
        stat_card(f"{ex['sentence_bleu']:.1f}", "This example's sentence BLEU"),
        stat_card(match_label, "Exact match (this example)", tone=tone),
    )

    # Outcome-dependent copy baked into the on-screen HTML itself (not just the spoken
    # script), so the framing survives even if the dashboard is viewed asynchronously
    # (shared, screenshotted) without the presenter's narration.
    if is_match:
        outcome_note = (
            "This row happens to be an exact match: the outcome about two-thirds of "
            "random draws land on."
        )
    else:
        outcome_note = (
            "This row is not an exact match: the outcome about one-third of random "
            "draws land on, expected and disclosed above, not a failure. Differences from "
            "the reference are often structural (e.g. a different but equally valid way of "
            "writing the same method), so there's no need to hunt for a line-by-line diff."
        )

    aggregate_html = note_card(
        "Aggregate over all 1,000 (same model, same beam=4, see Results tab)",
        f"Corpus BLEU {beam_beam4['corpus_bleu']:.2f}, Exact Match {beam_beam4['exact_match']:.1f}%. "
        "This one row is a single data point behind that number, not a new metric: corpus BLEU is "
        "computed over n-gram counts across the whole test set, not by averaging per-example "
        f"sentence BLEU scores like the one above. {outcome_note}",
    )
    return ex["cs"], ex["java_reference"], ex["java_prediction"], stats_html, aggregate_html


CUSTOM_CSS = """
html, body, gradio-app {
    background: #1a1a19 !important;
}
.gradio-container {
    background: #1a1a19 !important;
    color: #ffffff !important;
    width: 94% !important;
    max-width: 1500px !important;
    margin: auto !important;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container h4, .gradio-container .prose, .gradio-container .prose p,
.gradio-container .prose li, .gradio-container .md, .gradio-container span {
    color: #ffffff !important;
}
.gradio-container .prose em, .gradio-container .prose i,
.gradio-container .md em, .gradio-container .md i {
    color: #c3c2b7 !important;
}
.gradio-container .prose strong, .gradio-container .prose b {
    color: #ffffff !important;
}
button[role="tab"] {
    font-size: 1.02rem !important;
    padding: 10px 18px !important;
    color: #c3c2b7 !important;
}
button[role="tab"].selected {
    color: #e0b44c !important;
}
.stat-row, .note-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 6px 0 22px;
}
.stat-card, .note-card {
    flex: 1 1 160px;
    background: #232322;
    border: 1px solid #2c2c2a;
    border-radius: 10px;
    padding: 16px 18px;
}
.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
}
.stat-label {
    font-size: 0.82rem;
    color: #c3c2b7;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.stat-delta {
    display: inline-block;
    margin-top: 10px;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.stat-tone-good .stat-delta { background: rgba(12, 163, 12, 0.18); color: #3ddc3d; }
.stat-tone-bad .stat-delta { background: rgba(208, 59, 59, 0.18); color: #ff7a7a; }
.stat-tone-neutral .stat-delta { background: rgba(195, 194, 183, 0.15); color: #c3c2b7; }
.note-card { flex-basis: 220px; }
.note-title { font-weight: 700; color: #ffffff; margin-bottom: 6px; }
.note-body { font-size: 0.9rem; color: #c3c2b7; line-height: 1.45; }
.callout {
    background: #1f2a22;
    border-left: 4px solid #199e70;
    padding: 14px 18px;
    border-radius: 8px;
    margin: 4px 0 20px;
}
.callout-title { font-weight: 700; font-size: 1.05rem; color: #ffffff; margin-bottom: 4px; }
.callout-body { color: #c3c2b7; font-size: 0.92rem; }
/* Reserved for null/noise findings (e.g. "small vs base: too close to call"), so
   they don't share the same green "confirmed insight" accent as callouts about
   real, confirmed effects (fine-tuning, code-pretraining, beam search). */
.callout-muted {
    background: #232322;
    border-left-color: #6b6b66;
}
.gradio-container pre, .gradio-container pre code {
    background: #232322 !important;
    color: #e6e6e2 !important;
}
.gradio-container .token.keyword { color: #3987e5 !important; }
.gradio-container .token.class-name, .gradio-container .token.function { color: #86b6ef !important; }
.gradio-container .token.string, .gradio-container .token.builtin { color: #199e70 !important; }
.gradio-container .token.comment { color: #8a8a85 !important; }
.gradio-container .token.punctuation, .gradio-container .token.operator { color: #c3c2b7 !important; }
/* gr.Code (Example Browser tab) renders via CodeMirror, not <pre>/.token.* like the
   Markdown code fences above: without these rules its editor keeps Gradio's default
   light background while the broad "span" rule above still forces its text white,
   making the code panels unreadable (white-on-white). */
.gradio-container .cm-editor, .gradio-container .cm-editor .cm-scroller {
    background: #232322 !important;
}
.gradio-container .cm-editor .cm-content, .gradio-container .cm-editor .cm-line {
    color: #e6e6e2 !important;
}
.gradio-container .cm-editor .cm-gutters {
    background: #1f1f1e !important;
    color: #8a8a85 !important;
    border-color: #2c2c2a !important;
}
.gradio-container .cm-editor .cm-activeLine, .gradio-container .cm-editor .cm-activeLineGutter {
    background: rgba(255, 255, 255, 0.04) !important;
}
"""

with gr.Blocks(title="C# -> Java - CodeT5") as demo:
    gr.Markdown(
        "# From C# to Java with CodeT5\n"
        "Fine-tuning, evaluation, and an ablation study across two CodeT5 models and a "
        "non-code-pretrained baseline, on the CodeXGLUE dataset."
    )

    with gr.Tabs():
        with gr.Tab("Data"):
            gr.Markdown("### Dataset split")
            gr.HTML(stat_row(
                stat_card(f"{PHASE_A['train']:,}", "Train examples"),
                stat_card(f"{PHASE_A['validation']:,}", "Validation examples"),
                stat_card(f"{PHASE_A['test']:,}", "Test examples"),
            ))

        with gr.Tab("Training Setup"):
            gr.Markdown("### Hyperparameters")
            gr.HTML(stat_row(
                stat_card(PHASE_B["learning_rate"], "Learning rate"),
                stat_card(str(PHASE_B["batch_size"]), "Batch size"),
                stat_card(str(PHASE_B["epochs"]), "Epochs"),
                stat_card(str(PHASE_B["max_length"]), "Max length"),
            ))
            gr.Markdown(
                "Identical recipe for all three models (codet5-small, codet5-base, t5vanilla): any "
                "difference in results traces back to the model, not the setup."
            )
            gr.Markdown("### Training cost per model (measured, same Colab T4 GPU)")
            gr.HTML(stat_row(
                stat_card(f"{phase_b_training['small']['train_runtime'] / 60:.1f} min", "codet5-small"),
                stat_card(f"{phase_b_training['base']['train_runtime'] / 60:.1f} min", "codet5-base"),
                stat_card(f"{phase_b_training['t5vanilla']['train_runtime'] / 60:.1f} min", "t5vanilla",
                          delta=f"+{flos_pretrain_delta_pct:.0f}% FLOs vs codet5-base", tone="neutral"),
            ))
            gr.Markdown(
                "*t5vanilla: same size as codet5-base, but its generic tokenizer represents code "
                "less efficiently, so each step costs more compute.*"
            )
            gr.Markdown("### Validation loss per epoch, all three models")
            gr.Plot(
                plot_loss_curves([
                    {"curve": pretraining_loss_curves["small"], "color": SERIES_SMALL, "label": "codet5-small"},
                    {"curve": pretraining_loss_curves["base"], "color": SERIES_BASE, "label": "codet5-base"},
                    {"curve": pretraining_loss_curves["t5vanilla"], "color": SERIES_T5VANILLA, "label": "t5vanilla"},
                ]),
                show_label=False,
            )
            gr.Markdown(
                "*codet5-base overfits from epoch 3; t5vanilla is still improving at epoch 10 "
                "(not converged in this budget).*"
            )

        with gr.Tab("Results: Model Comparison"):
            gr.HTML(
                f'<div class="callout">'
                f'<div class="callout-title">Model selected: {BEST_FINETUNED_LABEL} (fine-tuned)</div>'
                f'<div class="callout-body">Highest corpus BLEU '
                f"({best_family['fine_tuned']['corpus_bleu']:.2f}) among all three fine-tuned models, "
                f"with Exact Match a hair below codet5-small's (a 0.6pp gap this run, not a reliable "
                f"difference: see below).</div></div>"
            )
            gr.Markdown("#### The effect of fine-tuning (BLEU, zero-shot -> fine-tuned)")
            gr.Plot(plot_dumbbell(phase_c["families"]), show_label=False)
            gr.Markdown(
                "*Here light/dark blue = zero-shot vs fine-tuned; in the next chart each model gets "
                "its own color.*"
            )
            gr.Markdown("#### small vs base vs t5vanilla, fine-tuned models")
            gr.Plot(plot_model_comparison(phase_c["families"]), show_label=False)
            gr.HTML(
                '<div class="callout callout-muted">'
                '<div class="callout-title">small vs base: too close to call</div>'
                '<div class="callout-body">Retraining from scratch flipped which model leads on which '
                'metric: the gap is smaller than run-to-run training noise, not a reliable difference '
                'either way.</div></div>'
            )
            gr.Markdown("#### Does code-specific pretraining matter? (t5vanilla vs codet5-base, same size)")
            gr.HTML(stat_row(
                stat_card(f"{t5vanilla_family['fine_tuned']['corpus_bleu']:.2f}", "t5vanilla BLEU",
                          delta=f"{-pretrain_bleu_delta:+.2f} pt vs codet5-base", tone="neutral"),
                stat_card(f"{t5vanilla_family['fine_tuned']['exact_match']:.1f}%", "t5vanilla Exact Match",
                          delta=f"{-pretrain_em_delta:+.1f} pp vs codet5-base", tone="bad"),
            ))
            gr.HTML(
                '<div class="callout">'
                '<div class="callout-title">Yes, and it shows up mostly on Exact Match</div>'
                '<div class="callout-body">Same size, same recipe, only the pretraining corpus differs '
                '(code vs generic text). On Exact Match the effect rivals fine-tuning itself, and dwarfs '
                'the small-vs-base gap above.</div></div>'
            )

        with gr.Tab("Results: Decoding & Errors"):
            gr.Markdown("#### Decoding strategy: greedy vs beam search (codet5-base, fine-tuned)")
            gr.Plot(plot_decoding_comparison(beam_search), show_label=False)
            gr.HTML(stat_row(
                stat_card(f"{beam_beam4['corpus_bleu']:.2f}", "BLEU, beam=4",
                          delta=f"{beam_bleu_delta:+.2f} pt vs greedy", tone="neutral"),
                stat_card(f"{beam_beam4['exact_match']:.1f}%", "Exact Match, beam=4",
                          delta=f"{beam_em_delta:+.1f} pp vs greedy", tone="neutral"),
                stat_card(f"{beam_beam4['elapsed_seconds']:.0f}s", "Generation time, beam=4",
                          delta=f"{beam_time_delta_pct:+.0f}% vs greedy", tone="neutral"),
            ))
            gr.HTML(
                '<div class="callout">'
                '<div class="callout-title">A small, direction-confirmed gain from beam search</div>'
                f'<div class="callout-body">Measured cost: {beam_time_delta_pct:+.0f}% wall-clock, far '
                'below the naive ~4x estimate (beams are batched on the GPU). The gain is ~two orders of '
                'magnitude smaller than fine-tuning and too small to call statistically confirmed, but '
                'cheap enough to keep.</div></div>'
            )
            gr.Markdown("#### Outcome composition per model")
            gr.Plot(plot_error_composition(error_rows), show_label=False)
            gr.Markdown(
                "#### An example from the test set\n\n"
                "```csharp\n"
                "// C# (input)\n"
                "public int CompareTo(object other) {\n"
                "    BytesRef br = other as BytesRef;\n"
                "    Debug.Assert(br != null);\n"
                "    return utf8SortedAsUnicodeSortOrder.Compare(this, br);\n"
                "}\n"
                "```\n\n"
                "```java\n"
                "// Java reference (human, from the test set)\n"
                "public int compareTo(BytesRef other) {\n"
                "    return Arrays.compareUnsigned(\n"
                "        this.bytes, this.offset, this.offset + this.length,\n"
                "        other.bytes, other.offset, other.offset + other.length\n"
                "    );\n"
                "}\n"
                "```\n\n"
                "```java\n"
                "// Java generated by the model (codet5-base, fine-tuned)\n"
                "public int compareTo(Object other) {\n"
                "    assert other instanceof BytesRef;\n"
                "    final BytesRef br = (BytesRef) other;\n"
                "    return br.compareTo(this);\n"
                "}\n"
                "```\n\n"
                "*The model translates the C# structure faithfully; the human reference solves the method "
                "a different way. A dataset limitation, not a model mistake.*"
            )

        with gr.Tab("Example Browser"):
            gr.Markdown(
                "### One precomputed prediction at a time\n"
                "**Nothing is generated live**: each click draws one already-computed example, "
                "uniformly at random from all 1,000 test rows (codet5-base fine-tuned, beam=4). "
                f"About 2 in 3 draws are exact matches ({beam_beam4['exact_match']:.1f}% of the test set)."
            )
            example_random_btn = gr.Button("Random example", variant="primary")
            with gr.Row():
                # language="cpp" on all three boxes below, not "csharp"/"java": gradio 6.20.0's
                # gr.Code hardcodes its allowed `language` values (see gradio/components/code.py)
                # and raises ValueError for anything outside {python, c, cpp, markdown, latex,
                # json, html, css, javascript, jinja2, typescript, yaml, dockerfile, shell, r,
                # sql(+dialects), None}: neither "java" nor "csharp" is in that list (verified
                # directly against the installed 6.20.0 source and at runtime). "cpp" is the
                # closest available curly-brace/semicolon language and won't crash the tab. Do
                # not "fix" this back to "java"/"csharp" without checking that list again.
                example_cs_code = gr.Code(
                    label="C# (input)",
                    language="cpp",
                    value='// Click "Random example" above to load one of the 1,000 test-set rows.',
                    interactive=False,
                    wrap_lines=True,
                    lines=10,
                    max_lines=14,
                    buttons=["copy"],
                )
                example_java_ref_code = gr.Code(
                    label="Java (human reference)",
                    language="cpp",
                    value="// ...",
                    interactive=False,
                    wrap_lines=True,
                    lines=10,
                    max_lines=14,
                    buttons=["copy"],
                )
                example_java_pred_code = gr.Code(
                    label="Java (model prediction, precomputed)",
                    language="cpp",
                    value="// ...",
                    interactive=False,
                    wrap_lines=True,
                    lines=10,
                    max_lines=14,
                    buttons=["copy"],
                )
            example_stats_html = gr.HTML()
            example_aggregate_html = gr.HTML()

            example_random_btn.click(
                fn=pick_random_example,
                inputs=None,
                outputs=[
                    example_cs_code,
                    example_java_ref_code,
                    example_java_pred_code,
                    example_stats_html,
                    example_aggregate_html,
                ],
                # This is a JSON lookup, not model inference: suppress Gradio's default
                # loading spinner so it can't be misread as "the model is running".
                show_progress="hidden",
            )

        with gr.Tab("Ablation Study (128 vs 256)"):
            gr.Markdown("#### Validation loss per epoch")
            gr.Plot(
                plot_loss_curves([
                    {"curve": phase_d["len256"]["eval_loss_curve"], "color": SEQ_DARK, "label": "max_length = 256"},
                    {"curve": phase_d["len128"]["eval_loss_curve"], "color": SEQ_LIGHT, "label": "max_length = 128"},
                ]),
                show_label=False,
            )
            gr.Markdown("#### Delta: 128 vs 256 tokens")
            gr.HTML(stat_row(
                ablation_card(f"{bleu_delta:+.2f} pt", "BLEU", good=bleu_delta > 0),
                ablation_card(f"{em_delta:+.1f} pp", "Exact Match", good=em_delta > 0),
                ablation_card(f"{time_delta_pct:+.1f}%", f"Training time ({train_minutes_128:.1f} vs {train_minutes_256:.1f} min)",
                              good=time_delta_pct < 0),
                ablation_card(f"{flos_delta_pct:+.1f}%", f"Total FLOs ({phase_d['len128']['total_flos']:.2e} vs {phase_d['len256']['total_flos']:.2e})",
                              good=flos_delta_pct < 0),
            ))
            gr.Markdown(
                "**Conclusion**: the compute savings at 128 tokens don't justify the quality drop "
                "(methods longer than 128 tokens get truncated). **256 remains the better choice.**"
            )
            gr.Markdown(
                "#### Wrapping up\n"
                "**Final configuration: codet5-base, fine-tuned, max_length 256, beam search.** "
                "Natural next steps: early stopping, stronger regularization, data cleaning."
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"), css=CUSTOM_CSS)
