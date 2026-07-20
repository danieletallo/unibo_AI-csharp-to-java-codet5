# DATA LOADING
#
# Reads Phase C and Phase D's saved JSON results and shapes them into the
# small structures the dashboard's tabs need. Phase A/B facts are static
# (recorded in report/report.txt, never saved as JSON) so they're plain
# constants here.
#
# NOTE: PHASE_A/PHASE_B stay as two independent dicts (not merged into one),
# so app.py can show them as two separate narrative beats, Data and then
# Training Setup, without touching this loading logic.

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_C_PATH = os.path.join(REPO_ROOT, "phase_C_evaluation", "results", "metrics.json")
ABLATION_D_PATH = os.path.join(REPO_ROOT, "phase_D_ablation", "results", "ablation_results.json")
BEAM_SEARCH_PATH = os.path.join(REPO_ROOT, "phase_C_evaluation", "results", "beam_search_results.json")
BEAM_SEARCH_PREDICTIONS_PATH = os.path.join(REPO_ROOT, "phase_C_evaluation", "results", "beam_search_predictions.json")
TRAIN_METRICS_DIR = os.path.join(REPO_ROOT, "phase_B_finetuning", "results")

PHASE_A = {"train": 10300, "validation": 500, "test": 1000}
PHASE_B = {"learning_rate": "5e-5", "batch_size": 8, "epochs": 10, "max_length": 256}

# Maps each model family to the file suffix used for its Phase B artifacts
# (train_metrics_<suffix>.json, trainer_state_<suffix>.json).
TRAIN_ARTIFACT_SUFFIX = {"small": "codet5_small", "base": "codet5_base", "t5vanilla": "t5vanilla"}


def load_phase_c():
    with open(METRICS_C_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    by_name = {m["name"]: m for m in payload["models"]}
    return {
        "best_finetuned_model": payload["best_finetuned_model"],
        "by_name": by_name,
        "families": [
            {
                "key": "small",
                "label": "codet5-small",
                "zero_shot": by_name["codet5-small (zero-shot)"],
                "fine_tuned": by_name["codet5-small (fine-tuned)"],
            },
            {
                "key": "base",
                "label": "codet5-base",
                "zero_shot": by_name["codet5-base (zero-shot)"],
                "fine_tuned": by_name["codet5-base (fine-tuned)"],
            },
            {
                "key": "t5vanilla",
                "label": "t5vanilla",
                "zero_shot": by_name["t5vanilla (zero-shot)"],
                "fine_tuned": by_name["t5vanilla (fine-tuned)"],
            },
        ],
    }


def load_phase_b_training():
    """Per-model training cost (Phase B): wall-clock train_runtime and total_flos,
    read from the train_metrics_<family>.json files saved by finetune_models.py
    (the Trainer.train() return value, not recomputed here).
    """
    training = {}
    for family, suffix in TRAIN_ARTIFACT_SUFFIX.items():
        path = os.path.join(TRAIN_METRICS_DIR, f"train_metrics_{suffix}.json")
        with open(path, "r", encoding="utf-8") as f:
            training[family] = json.load(f)
    return training


def load_pretraining_loss_curves():
    """Per-model validation-loss curve (Phase B): one point per epoch, extracted from
    the trainer_state_<family>.json log_history (the same file the Trainer saves on
    its own; only the entries carrying an eval_loss are kept here).
    """
    curves = {}
    for family, suffix in TRAIN_ARTIFACT_SUFFIX.items():
        path = os.path.join(TRAIN_METRICS_DIR, f"trainer_state_{suffix}.json")
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        curves[family] = [
            {"epoch": entry["epoch"], "eval_loss": entry["eval_loss"]}
            for entry in state["log_history"]
            if "eval_loss" in entry
        ]
    return curves


def load_phase_d():
    with open(ABLATION_D_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    by_len = {r["max_length"]: r for r in results}
    return {"len256": by_len[256], "len128": by_len[128]}


def load_beam_search():
    """Greedy vs beam-search (num_beams=4) decoding comparison on the chosen fine-tuned
    model (codet5-base). Keyed by num_beams, mirroring load_phase_d's by-max_length
    shape, so callers index it the same way: beam_search["by_beams"][1] / [4].
    """
    with open(BEAM_SEARCH_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    by_beams = {r["num_beams"]: r for r in payload["results"]}
    return {"model": payload["model"], "by_beams": by_beams}


def load_beam_search_predictions():
    """Per-example beam-search (num_beams=4) predictions for the chosen fine-tuned
    model (codet5-base): 1,000 rows, the same test set and configuration that
    produced the aggregate corpus BLEU (89.81) and Exact Match (66.6%) already
    shown via load_beam_search() in the Results tab. Powers the Example Browser
    tab's random-lookup button; nothing here is computed at dashboard runtime,
    this just reads the JSON Phase C already saved after beam_search_comparison.py ran.

    Each row: {index, cs, java_reference, java_prediction, sentence_bleu, exact_match}.
    NOTE: sentence_bleu is a per-example BLEU score. Averaging this column across
    rows is NOT the same figure as the corpus BLEU shown in Results (corpus BLEU
    aggregates n-gram counts across the whole test set; averaging per-sentence
    scores is a different, typically higher-reading statistic); callers must
    never present a mean of this column as if it were corpus BLEU.
    """
    with open(BEAM_SEARCH_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "model": payload["model"],
        "num_beams": payload["num_beams"],
        "examples": payload["examples"],
    }
