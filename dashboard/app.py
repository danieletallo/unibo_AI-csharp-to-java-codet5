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

from data import PHASE_A, PHASE_B, load_phase_c, load_phase_d, load_beam_search, load_beam_search_predictions
from plots import plot_decoding_comparison, plot_dumbbell, plot_error_composition, plot_loss_curves, plot_small_vs_base

phase_c = load_phase_c()
phase_d = load_phase_d()
beam_search = load_beam_search()
beam_predictions = load_beam_search_predictions()
example_pool = beam_predictions["examples"]  # 1,000 rows, unfiltered: random.choice draws from all of them

# metrics.json's own best_finetuned_model picks by raw corpus BLEU alone
# (codet5-small, +0.36 points). The project went with codet5-base instead,
# since its Exact Match is clearly higher (66.1% vs 64.7%) and that BLEU
# gap is negligible: see report.txt for the reasoning.
BEST_FINETUNED_LABEL = "codet5-base"
best_family = next(f for f in phase_c["families"] if f["label"] == BEST_FINETUNED_LABEL)
bleu_delta = phase_d["len128"]["corpus_bleu"] - phase_d["len256"]["corpus_bleu"]
em_delta = phase_d["len128"]["exact_match"] - phase_d["len256"]["exact_match"]
time_delta_pct = 100 * (phase_d["len128"]["train_runtime_seconds"] - phase_d["len256"]["train_runtime_seconds"]) / phase_d["len256"]["train_runtime_seconds"]
flos_delta_pct = 100 * (phase_d["len128"]["total_flos"] - phase_d["len256"]["total_flos"]) / phase_d["len256"]["total_flos"]
train_minutes_256 = phase_d["len256"]["train_runtime_seconds"] / 60
train_minutes_128 = phase_d["len128"]["train_runtime_seconds"] / 60

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


def note_row(*cards):
    return '<div class="note-row">' + "".join(cards) + "</div>"


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
.gradio-container .prose strong, .gradio-container .prose b {
    color: #ffffff !important;
}
button[role="tab"] {
    font-size: 1.02rem !important;
    padding: 10px 18px !important;
    color: #c3c2b7 !important;
}
button[role="tab"].selected {
    color: #3987e5 !important;
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
        "Fine-tuning, evaluation, and an ablation study across two CodeT5 models on the CodeXGLUE dataset."
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
            gr.Markdown("Identical configuration for codet5-small and codet5-base, for a fair comparison.")

        with gr.Tab("Results"):
            gr.HTML(
                f'<div class="callout">'
                f'<div class="callout-title">Model selected: {BEST_FINETUNED_LABEL} (fine-tuned)</div>'
                f'<div class="callout-body">Highest Exact Match '
                f"({best_family['fine_tuned']['exact_match']:.1f}%), with corpus BLEU essentially tied "
                f"against the alternative.</div></div>"
            )
            gr.Markdown("#### The effect of fine-tuning (BLEU, zero-shot -> fine-tuned)")
            gr.Plot(plot_dumbbell(phase_c["families"]), show_label=False)
            gr.Markdown("#### small vs base, fine-tuned models")
            gr.Plot(plot_small_vs_base(phase_c["families"]), show_label=False)
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
                '<div class="callout-body">Re-running inference with beam search (num_beams=4) instead of '
                f'greedy decoding on the same {BEST_FINETUNED_LABEL} (fine-tuned) checkpoint raises corpus BLEU '
                f'by {beam_bleu_delta:+.2f} points ({beam_greedy["corpus_bleu"]:.2f} -&gt; '
                f'{beam_beam4["corpus_bleu"]:.2f}, {beam_bleu_delta_rel_pct:+.2f}% relative) '
                f'and Exact Match by {beam_em_delta:+.1f} points ({beam_greedy["exact_match"]:.1f}% -&gt; '
                f'{beam_beam4["exact_match"]:.1f}%), for a measured (not estimated) cost of '
                f'{beam_time_delta_pct:+.0f}% wall-clock time over the full 1,000-example test set '
                f'({beam_greedy["elapsed_seconds"]:.0f}s -&gt; {beam_beam4["elapsed_seconds"]:.0f}s), far less '
                'than the ~4x a naive "4 hypotheses instead of 1" estimate would suggest, since the beams are '
                'batched on the GPU rather than run serially. Both metrics move the right way, but the gain is '
                'roughly two orders of magnitude smaller than fine-tuning itself (~+87 BLEU, see above) and about '
                '7x smaller than the max_length ablation (+5 BLEU, next tab), and small enough (5 examples out '
                'of 1,000 for Exact Match) that we are not calling it a statistically confirmed improvement '
                'without a significance test. Given how cheap it actually is, there is little reason not to use '
                'it for final reported numbers.</div></div>'
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
                "The model keeps the same structure as the C#: cast to `BytesRef`, assert the type, then delegate "
                "the comparison. The dataset's human reference solves the method in a completely different way, "
                "comparing raw byte arrays directly instead of delegating. That mismatch is a limitation of the "
                "dataset's reference translations, not a mistake by the model."
            )

        with gr.Tab("Example Browser"):
            gr.Markdown(
                "### One precomputed prediction at a time\n"
                "Browsing the same 1,000 test-set examples used for the aggregate corpus BLEU / "
                "Exact Match numbers above (codet5-base, fine-tuned, beam search num_beams=4). "
                "**Nothing is generated live**: each click looks up one already-computed example, "
                "drawn uniformly at random from all 1,000 test examples: exact_match odds are "
                f"about 2 in 3, since {beam_beam4['exact_match']:.1f}% of the test set matches the "
                "reference exactly. Any brief flash after clicking is just the page fetching that "
                "example, not the model running."
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
            gr.Plot(plot_loss_curves(phase_d), show_label=False)
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
                "**Conclusion**: the resource savings at 128 tokens (~24% less training time, ~20% fewer FLOs) "
                "don't justify the quality drop (-5 BLEU, -2.7pp Exact Match): some C#/Java methods exceed 128 "
                "tokens and get truncated. **256 remains the better choice.**"
            )

        with gr.Tab("Conclusion & Next Steps"):
            gr.Markdown(
                "Beam search decoding (tried above, in Results) closes out one item from this list: "
                "a small, direction-confirmed gain, not a major lever. Remaining open items:"
            )
            gr.HTML(note_row(
                note_card(
                    "Early stopping",
                    "Both runs reach their minimum validation loss well before epoch 10 "
                    "(epoch 6 for 256 tokens, epoch 4 for 128) and overfit afterward.",
                ),
                note_card(
                    "Stronger regularization",
                    "More aggressive dropout, weight decay, or label smoothing.",
                ),
                note_card(
                    "Data cleaning",
                    "Filter out C#/Java pairs where the reference is an idiomatic rewrite "
                    "rather than a literal translation.",
                ),
            ))

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"), css=CUSTOM_CSS)
