"""
Phase C extra -- Beam search vs greedy decoding, for the chosen model only.

Compares greedy decoding (num_beams=1) against beam search (num_beams=4) on
codet5-base (fine-tuned): the model the project actually goes with. Beam
search cost scales with num_beams, so there's no reason to pay it for
small or zero-shot, which aren't the final choice. Greedy is re-run here
too, rather than reusing metrics.json's saved numbers, so both sides of
the comparison come from the exact same run and are directly comparable.

Saved as its own JSON file, separate from metrics.json.
"""

import os

# Disables TensorFlow auto-detection in transformers -- see Phase C for the
# torch/torchvision conflict this avoids on Colab.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import sys
import time
import traceback

import evaluate
import torch

from phase_A_preparation.prepare_dataset import load_and_prepare_dataset
from phase_C_evaluation.evaluate_models import evaluate_model, get_device, load_model_and_tokenizer

# Redefined here instead of imported from evaluate_models, since that module
# has no __main__ guard and importing it would re-run Phase C's evaluation.
DRIVE_DIR = "/content/drive/MyDrive/csharp_to_java_project"

MODEL_PATH = f"{DRIVE_DIR}/fine_tuned_codet5_base"
MODEL_NAME = "codet5-base (fine-tuned)"
BEAM_WIDTHS = [1, 4]


def run_beam_search_comparison():
    print("Step 1/3: loading and preparing the test dataset...", flush=True)
    prepared_dataset = load_and_prepare_dataset()
    source_texts = prepared_dataset["test"]["cs"]
    reference_texts = prepared_dataset["test"]["java"]
    print(f"Test set ready: {len(source_texts)} examples.", flush=True)

    device = get_device()
    print(f"Step 2/3: using device '{device}', loading '{MODEL_PATH}'...", flush=True)
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    bleu_metric = evaluate.load("sacrebleu")

    print("Step 3/3: generating with each beam width...", flush=True)
    results = []
    for num_beams in BEAM_WIDTHS:
        label = "greedy" if num_beams == 1 else f"beam search (num_beams={num_beams})"
        print(f"[{label}] generating translations for {len(source_texts)} examples...", flush=True)
        start_time = time.time()
        evaluation = evaluate_model(
            MODEL_NAME, model, tokenizer, source_texts, reference_texts, device, bleu_metric,
            num_beams=num_beams,
        )
        elapsed_seconds = time.time() - start_time
        print(f"[{label}] corpus BLEU: {evaluation['corpus_bleu']:.2f}, exact match: {evaluation['exact_match']:.2f}%, "
              f"took {elapsed_seconds:.1f}s", flush=True)
        results.append({
            "num_beams": num_beams,
            "label": label,
            "corpus_bleu": evaluation["corpus_bleu"],
            "exact_match": evaluation["exact_match"],
            "elapsed_seconds": elapsed_seconds,
        })

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def print_comparison(results):
    print("=== Beam search vs greedy, codet5-base (fine-tuned) ===")
    print(f"{'num_beams':<12}{'corpus BLEU':>15}{'exact match %':>16}{'time (s)':>12}")
    for result in results:
        print(f"{result['num_beams']:<12}{result['corpus_bleu']:>15.2f}{result['exact_match']:>16.2f}{result['elapsed_seconds']:>12.1f}")


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "beam_search_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "model_path": MODEL_PATH, "results": results}, f, indent=2)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    try:
        comparison_results = run_beam_search_comparison()
        print_comparison(comparison_results)
        save_results(comparison_results, f"{DRIVE_DIR}/phase_C_results")
        print("Done.", flush=True)
    except Exception:
        print("Beam search comparison crashed, full traceback below:", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
