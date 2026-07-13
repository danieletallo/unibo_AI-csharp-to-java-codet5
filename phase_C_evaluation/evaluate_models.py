"""
Phase C -- Model evaluation.

Generates the Java translation for the test set examples with both
models fine-tuned in Phase B, evaluates performance with appropriate
metrics (BLEU, possibly Exact Match), analyzes some recurring errors,
and compares the two models, highlighting their strengths and
weaknesses.

Also compares each fine-tuned model against its own zero-shot baseline
(the same pretrained checkpoint, without any fine-tuning), to check
whether fine-tuning is actually adding value on this task.
"""

import os

# Useful for running in Colab, where TensorFlow is installed by default and
# can conflict with PyTorch. Disables TF and its logging, so only PyTorch
# and Hugging Face Transformers are used. 
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import re
import sys
import traceback

import evaluate
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from phase_A_preparation.prepare_dataset import load_and_prepare_dataset

# Redefined here instead of imported from phase_B_finetuning.finetune_models,
# since that module has no __main__ guard and importing it would re-run fine-tuning.
DRIVE_DIR = "/content/drive/MyDrive/csharp_to_java_project"

MODEL_CONFIGS = [
    {"name": "codet5-small (fine-tuned)", "model_path": f"{DRIVE_DIR}/fine_tuned_codet5_small", "family": "small", "is_finetuned": True},
    {"name": "codet5-base (fine-tuned)", "model_path": f"{DRIVE_DIR}/fine_tuned_codet5_base", "family": "base", "is_finetuned": True},
    {"name": "codet5-small (zero-shot)", "model_path": "Salesforce/codet5-small", "family": "small", "is_finetuned": False},
    {"name": "codet5-base (zero-shot)", "model_path": "Salesforce/codet5-base", "family": "base", "is_finetuned": False},
]


def normalize_code_text(text):
    """Strips and collapses whitespace, so only whitespace differences are ignored by Exact Match."""
    return re.sub(r"\s+", " ", text.strip())


def get_device():
    """Returns 'cuda' if a GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_path):
    """Loads a model + tokenizer from a local Drive path or a Hugging Face Hub id."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(get_device())
    model.eval()
    return model, tokenizer


def generate_predictions(model, tokenizer, source_texts, device, batch_size=8, max_length=256, num_beams=1):
    """Runs batched generation over source_texts and returns the decoded predictions."""
    predictions = []
    with torch.no_grad():
        for start in range(0, len(source_texts), batch_size):
            batch = source_texts[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
            predictions.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    return predictions


def compute_corpus_bleu(predictions, references, bleu_metric):
    """Computes corpus-level BLEU."""
    result = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    return result["score"]


def compute_sentence_bleu(prediction, reference, bleu_metric):
    """Computes BLEU for a single prediction/reference pair, used to rank examples for error analysis."""
    result = bleu_metric.compute(predictions=[prediction], references=[[reference]])
    return result["score"]


def compute_exact_match(predictions, references):
    """Percentage of predictions that match their reference exactly after whitespace normalization."""
    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if normalize_code_text(pred) == normalize_code_text(ref)
    )
    return 100.0 * matches / len(predictions)


def evaluate_model(name, model, tokenizer, source_texts, reference_texts, device, bleu_metric, batch_size=8, max_length=256, num_beams=1):
    """Generates predictions for one model and scores them."""
    predictions = generate_predictions(model, tokenizer, source_texts, device, batch_size, max_length, num_beams)
    sentence_bleus = [compute_sentence_bleu(pred, ref, bleu_metric) for pred, ref in zip(predictions, reference_texts)]
    return {
        "name": name,
        "predictions": predictions,
        "sentence_bleus": sentence_bleus,
        "corpus_bleu": compute_corpus_bleu(predictions, reference_texts, bleu_metric),
        "exact_match": compute_exact_match(predictions, reference_texts),
    }


def find_worst_examples(source_texts, reference_texts, predictions, sentence_bleus, n=5):
    """Returns the n examples with the lowest sentence-level BLEU, for qualitative error analysis."""
    examples = list(zip(source_texts, reference_texts, predictions, sentence_bleus))
    examples.sort(key=lambda example: example[3])
    return examples[:n]


def print_worst_examples(model_name, worst_examples):
    """Prints full source/reference/prediction triples for the worst examples."""
    print(f"=== Worst {len(worst_examples)} examples for {model_name} ===")
    for source, reference, prediction, bleu in worst_examples:
        print(f"--- sentence BLEU: {bleu:.2f} ---")
        print(f"cs:        {source}")
        print(f"java (ref):{reference}")
        print(f"java (pred):{prediction}")


def categorize_common_errors(predictions, references, sentence_bleus, max_length, near_miss_bleu_threshold=70):
    """
    Counts recurring error categories per model: empty output, likely
    truncation, near-misses, and everything else. Truncation is approximated
    by checking if the decoded prediction's word count is close to
    max_length -- a lightweight heuristic, not an exact token count.
    """
    counts = {"empty_output": 0, "likely_truncated": 0, "near_miss": 0, "other_mismatch": 0}
    for prediction, reference, bleu in zip(predictions, references, sentence_bleus):
        normalized_pred = normalize_code_text(prediction)
        normalized_ref = normalize_code_text(reference)
        if not normalized_pred:
            counts["empty_output"] += 1
        elif normalized_pred == normalized_ref:
            continue
        elif len(prediction.split()) >= max_length - 2:
            counts["likely_truncated"] += 1
        elif bleu >= near_miss_bleu_threshold:
            counts["near_miss"] += 1
        else:
            counts["other_mismatch"] += 1
    return counts


def compare_models(results):
    """Prints a BLEU/Exact Match comparison table and returns the best fine-tuned model."""
    print("=== Model comparison ===")
    print(f"{'model':<30}{'corpus BLEU':>15}{'exact match %':>16}")
    for result in results:
        print(f"{result['name']:<30}{result['corpus_bleu']:>15.2f}{result['exact_match']:>16.2f}")

    finetuned_results = [result for result in results if result["is_finetuned"]]
    best = max(finetuned_results, key=lambda result: result["corpus_bleu"])
    print(f"Best fine-tuned model (by corpus BLEU): {best['name']}")
    return best


def compare_finetuned_vs_zeroshot(results):
    """Pairs each fine-tuned result with its same-family zero-shot baseline and prints the delta."""
    print("=== Fine-tuning impact (fine-tuned vs zero-shot, same family) ===")
    by_family = {}
    for result in results:
        by_family.setdefault(result["family"], {})[result["is_finetuned"]] = result

    for family, pair in by_family.items():
        finetuned, zero_shot = pair[True], pair[False]
        bleu_delta = finetuned["corpus_bleu"] - zero_shot["corpus_bleu"]
        em_delta = finetuned["exact_match"] - zero_shot["exact_match"]
        print(f"{family}: BLEU {zero_shot['corpus_bleu']:.2f} -> {finetuned['corpus_bleu']:.2f} (delta {bleu_delta:+.2f}), "
              f"Exact Match {zero_shot['exact_match']:.2f} -> {finetuned['exact_match']:.2f} (delta {em_delta:+.2f})")


def save_results(results, best_finetuned_name, output_dir):
    """Saves per-model metrics and worst examples, plus the best fine-tuned model's name, as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "best_finetuned_model": best_finetuned_name,
        "models": [
            {
                "name": result["name"],
                "model_path": result["model_path"],
                "corpus_bleu": result["corpus_bleu"],
                "exact_match": result["exact_match"],
                "error_categories": result["error_categories"],
                "worst_examples": [
                    {"cs": source, "java_reference": reference, "java_prediction": prediction, "sentence_bleu": bleu}
                    for source, reference, prediction, bleu in result["worst_examples"]
                ],
            }
            for result in results
        ],
    }
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {output_path}")


def run_evaluation():
    print("Step 1/4: loading and preparing the test dataset...", flush=True)
    prepared_dataset = load_and_prepare_dataset()
    source_texts = prepared_dataset["test"]["cs"]
    reference_texts = prepared_dataset["test"]["java"]
    print(f"Test set ready: {len(source_texts)} examples.", flush=True)

    device = get_device()
    print(f"Step 2/4: using device '{device}'.", flush=True)

    print("Step 3/4: loading the sacrebleu metric...", flush=True)
    bleu_metric = evaluate.load("sacrebleu")

    print("Step 4/4: evaluating each model in MODEL_CONFIGS...", flush=True)
    results = []
    for config in MODEL_CONFIGS:
        print(f"[{config['name']}] loading model from '{config['model_path']}'...", flush=True)
        model, tokenizer = load_model_and_tokenizer(config["model_path"])

        print(f"[{config['name']}] model loaded, generating translations for {len(source_texts)} examples...", flush=True)
        evaluation = evaluate_model(
            config["name"], model, tokenizer, source_texts, reference_texts, device, bleu_metric,
        )
        print(f"[{config['name']}] corpus BLEU: {evaluation['corpus_bleu']:.2f}, exact match: {evaluation['exact_match']:.2f}%", flush=True)

        worst_examples = find_worst_examples(source_texts, reference_texts, evaluation["predictions"], evaluation["sentence_bleus"])
        print_worst_examples(config["name"], worst_examples)
        error_categories = categorize_common_errors(evaluation["predictions"], reference_texts, evaluation["sentence_bleus"], max_length=256)

        results.append({
            **config,
            "corpus_bleu": evaluation["corpus_bleu"],
            "exact_match": evaluation["exact_match"],
            "worst_examples": worst_examples,
            "error_categories": error_categories,
        })

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_finetuned = compare_models(results)
    compare_finetuned_vs_zeroshot(results)
    save_results(results, best_finetuned["name"], f"{DRIVE_DIR}/phase_C_results")
    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception:
        print("Phase C evaluation crashed -- full traceback below:", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
