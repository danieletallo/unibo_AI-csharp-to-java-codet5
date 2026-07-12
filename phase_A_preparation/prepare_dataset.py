"""
Phase A -- Data preparation for C# -> Java code translation.

Loads google/code_x_glue_cc_code_to_code_trans, verifies and cleans it,
and exposes a reusable 'preprocess_function' for tokenization. The actual
tokenizer is supplied later (Phase B), once a model has been selected --
this module never imports or instantiates one.
"""

from datasets import load_dataset, DatasetDict

DATASET_NAME = "google/code_x_glue_cc_code_to_code_trans"
SPLITS = ("train", "validation", "test")


def load_raw_dataset(dataset_name=DATASET_NAME):
    """Loads the dataset with its predefined train/validation/test splits."""
    return load_dataset(dataset_name)


def print_dataset_overview(dataset_dict):
    """Prints column names, features, and example counts for each split."""
    print("=== Dataset overview ===")
    for split in SPLITS:
        ds = dataset_dict[split]
        print(f"[{split}] columns={ds.column_names} features={ds.features} num_examples={len(ds)}")


def show_sample_pairs(dataset_dict, split="train", n=3):
    """Prints n raw cs/java pairs from the given split for manual inspection."""
    print(f"=== Sample pairs from '{split}' ===")
    ds = dataset_dict[split]
    for i in range(min(n, len(ds))):
        example = ds[i]
        print(f"--- example {i} ---")
        print(f"cs:   {example['cs']}")
        print(f"java: {example['java']}")


def count_missing_or_empty(dataset_dict):
    """Counts rows per split where cs or java is missing/empty after stripping."""
    counts = {}
    for split in SPLITS:
        ds = dataset_dict[split]
        missing_cs = sum(1 for row in ds if not row["cs"] or not row["cs"].strip())
        missing_java = sum(1 for row in ds if not row["java"] or not row["java"].strip())
        counts[split] = {"missing_cs": missing_cs, "missing_java": missing_java}
    print(f"=== Missing/empty value counts === {counts}")
    return counts


def clean_dataset(dataset_dict, missing_counts):
    """Filters out rows with missing/empty cs or java, only if any were found."""
    needs_cleaning = any(
        counts["missing_cs"] or counts["missing_java"] for counts in missing_counts.values()
    )
    if not needs_cleaning:
        print("No missing/empty cs or java values found -- skipping cleaning.")
        return dataset_dict

    cleaned = {}
    for split in SPLITS:
        ds = dataset_dict[split]
        before = len(ds)
        cleaned_ds = ds.filter(
            lambda row: bool(row["cs"] and row["cs"].strip()) and bool(row["java"] and row["java"].strip())
        )
        after = len(cleaned_ds)
        print(f"[{split}] cleaned {before} -> {after} examples")
        cleaned[split] = cleaned_ds

    return DatasetDict(cleaned)


def load_and_prepare_dataset(dataset_name=DATASET_NAME):
    """Loads, inspects, and cleans the dataset. Convenience entry point for Phase B."""
    dataset_dict = load_raw_dataset(dataset_name)
    print_dataset_overview(dataset_dict)
    missing_counts = count_missing_or_empty(dataset_dict)
    return clean_dataset(dataset_dict, missing_counts)


def preprocess_function(examples, tokenizer, max_input_length=256, max_target_length=256, truncation=True):
    """
    Reusable tokenization function for dataset.map(..., batched=True).

    Tokenizes the 'cs' column as source input and the 'java' column as target
    output. The tokenizer is always supplied by the caller (Phase B) -- this
    function never selects or loads one itself.
    """
    model_inputs = tokenizer(examples["cs"], max_length=max_input_length, truncation=truncation)
    labels = tokenizer(text_target=examples["java"], max_length=max_target_length, truncation=truncation)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    prepared = load_and_prepare_dataset()
    show_sample_pairs(prepared, split="train", n=3)