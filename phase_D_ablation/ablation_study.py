"""
Phase D -- Ablation study and improvement.

Takes the best model resulting from Phase C (codet5-base) and conducts an
ablation study on the maximum sequence length (128 vs 256 tokens),
comparing translation quality, training time, and compute cost (total
FLOs). Peak GPU memory is left out of the comparison: it was never
measured for the 256 baseline in Phase B, so there is nothing to compare
the 128 run's measurement against.
Proposes possible improvements (e.g. data adaptation, other
regularization techniques).
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
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from phase_A_preparation.prepare_dataset import load_and_prepare_dataset, preprocess_function
from phase_C_evaluation.evaluate_models import get_device, generate_predictions, compute_corpus_bleu, compute_exact_match

DRIVE_DIR = "/content/drive/MyDrive/csharp_to_java_project"
BEST_MODEL_NAME = "Salesforce/codet5-base"
BEST_MODEL_METRICS_NAME = "codet5-base (fine-tuned)"
PHASE_C_METRICS_PATH = f"{DRIVE_DIR}/phase_C_results/metrics.json"

# Checked-in evidence for the max_length=256 baseline, already trained in
# Phase B: trainer_state.json is the real file saved by that Trainer run
# (has no train_runtime -- Trainer logs that summary to stdout without ever
# persisting it to trainer_state.json); console_log.json is that stdout
# summary, captured manually since no file records it otherwise.
BASELINE_256_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "baseline_256")


def extract_eval_loss_curve(log_history):
    """Extracts {epoch, eval_loss} pairs from a Trainer's log_history, for convergence comparisons."""
    return [{"epoch": entry["epoch"], "eval_loss": entry["eval_loss"]} for entry in log_history if "eval_loss" in entry]


def load_baseline_256_training_facts():
    """Reads training-time and compute-cost facts for the 256 baseline from the checked-in evidence files."""
    with open(os.path.join(BASELINE_256_ARTIFACTS_DIR, "console_log.json"), "r", encoding="utf-8") as f:
        console_log = json.load(f)
    with open(os.path.join(BASELINE_256_ARTIFACTS_DIR, "trainer_state.json"), "r", encoding="utf-8") as f:
        trainer_state = json.load(f)
    return {
        "train_runtime_seconds": console_log["train_runtime"],
        "total_flos": trainer_state["total_flos"],
        "eval_loss_curve": extract_eval_loss_curve(trainer_state["log_history"]),
    }


def load_quality_metrics(model_name, metrics_path):
    """Reads corpus_bleu and exact_match for model_name from Phase C's saved metrics.json."""
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for model_result in payload["models"]:
        if model_result["name"] == model_name:
            return {"corpus_bleu": model_result["corpus_bleu"], "exact_match": model_result["exact_match"]}
    raise ValueError(f"No entry named '{model_name}' found in {metrics_path}")


def build_trainer(model, tokenizer, tokenized_dataset, output_dir):
    """Builds a Seq2SeqTrainer with the same settings used in Phase B, only output_dir differs."""
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        load_best_model_at_end=True,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    return trainer


def train_with_max_length(max_length, prepared_dataset):
    """Fine-tunes a fresh codet5-base checkpoint at the given max_length, timing training and tracking compute cost."""
    print(f"Training codet5-base with max_length={max_length}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BEST_MODEL_NAME)

    tokenized_dataset = prepared_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer, "max_input_length": max_length, "max_target_length": max_length},
        batched=True,
    )

    output_dir = f"{DRIVE_DIR}/results_codet5_base_len{max_length}"
    trainer = build_trainer(model, tokenizer, tokenized_dataset, output_dir)

    start_time = time.perf_counter()
    trainer.train()
    train_runtime_seconds = time.perf_counter() - start_time
    total_flos = trainer.state.total_flos
    eval_loss_curve = extract_eval_loss_curve(trainer.state.log_history)

    fine_tuned_dir = f"{DRIVE_DIR}/fine_tuned_codet5_base_len{max_length}"
    trainer.save_model(fine_tuned_dir)
    tokenizer.save_pretrained(fine_tuned_dir)

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "max_length": max_length,
        "model_path": fine_tuned_dir,
        "train_runtime_seconds": train_runtime_seconds,
        "total_flos": total_flos,
        "eval_loss_curve": eval_loss_curve,
    }


def evaluate_checkpoint(model_path, max_length, source_texts, reference_texts, device, bleu_metric):
    """Generates translations for one checkpoint and scores them, reusing Phase C's generation and metric functions."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    predictions = generate_predictions(model, tokenizer, source_texts, device, max_length=max_length)
    corpus_bleu = compute_corpus_bleu(predictions, reference_texts, bleu_metric)
    exact_match = compute_exact_match(predictions, reference_texts)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"corpus_bleu": corpus_bleu, "exact_match": exact_match}


def print_comparison(results):
    """Prints a table comparing quality, training time, and compute cost across sequence lengths."""
    print("=== Ablation: max sequence length 128 vs 256 ===")
    print(f"{'max_length':<12}{'corpus BLEU':>15}{'exact match %':>16}{'train time (min)':>20}{'total FLOs':>16}")
    for result in results:
        train_minutes = result["train_runtime_seconds"] / 60
        flos = f"{result['total_flos']:.3e}"
        print(f"{result['max_length']:<12}{result['corpus_bleu']:>15.2f}{result['exact_match']:>16.2f}{train_minutes:>20.1f}{flos:>16}")


def save_results(results, output_dir):
    """Saves the ablation comparison as JSON, including per-epoch eval_loss curves for a dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ablation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")


def run_ablation():
    print("Step 1/3: loading and preparing the dataset...", flush=True)
    prepared_dataset = load_and_prepare_dataset()
    source_texts = prepared_dataset["test"]["cs"]
    reference_texts = prepared_dataset["test"]["java"]

    device = get_device()
    bleu_metric = evaluate.load("sacrebleu")

    print("Step 2/3: training codet5-base at max_length=128...", flush=True)
    training_result_128 = train_with_max_length(128, prepared_dataset)

    print("Evaluating the max_length=128 checkpoint on the test set...", flush=True)
    evaluation_128 = evaluate_checkpoint(training_result_128["model_path"], 128, source_texts, reference_texts, device, bleu_metric)
    result_128 = {**training_result_128, **evaluation_128}

    print("Step 3/3: comparing against the max_length=256 baseline from Phase B/C...", flush=True)
    baseline_256 = {
        "max_length": 256,
        "model_path": f"{DRIVE_DIR}/fine_tuned_codet5_base",
        **load_baseline_256_training_facts(),
        **load_quality_metrics(BEST_MODEL_METRICS_NAME, PHASE_C_METRICS_PATH),
    }
    results = [baseline_256, result_128]
    print_comparison(results)
    save_results(results, f"{DRIVE_DIR}/phase_D_results")
    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        run_ablation()
    except Exception:
        print("Phase D ablation study crashed -- full traceback below:", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
