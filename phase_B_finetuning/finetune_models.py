"""
Phase B -- Model fine-tuning.

Loads pretrained tokenizers and models (Salesforce/codet5-small and
Salesforce/codet5-base), tokenizes the dataset prepared in Phase A with
preprocess_function, and fine-tunes both models on the training set,
using the validation set to select the best checkpoint.
Records the training parameters used (learning rate, batch size, number
of epochs, maximum length).
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from phase_A_preparation.prepare_dataset import load_and_prepare_dataset, preprocess_function

# All checkpoints and fine-tuned models are saved under Google Drive instead of
# the Colab session's local disk, so nothing is lost if the session disconnects.
# Assumes drive.mount("/content/drive") was already run in the notebook.
DRIVE_DIR = "/content/drive/MyDrive/csharp_to_java_project"

# Loading tokenizer and models from Hugging Face Transformers (codet5-small and codet5-base):
# codet5-small
tokenizer_small = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
model_small = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

# codet5-base
tokenizer_base = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model_base = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")

# Fine-tuning the models on the dataset prepared in Phase A and tokenized with preprocess_function
# The dataset is loaded and prepared, then tokenized for both models
prepared_dataset = load_and_prepare_dataset()

tokenized_small = prepared_dataset.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer_small}, batched=True)
tokenized_base = prepared_dataset.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer_base}, batched=True)

def build_trainer(model, tokenizer, tokenized_dataset, output_dir):
    """
    Builds a Seq2SeqTrainer ready to fine-tune 'model' on 'tokenized_dataset'.

    Same settings are used for every model passed in, since Phase B asks to
    keep configurations as similar as possible across models -- only the
    model/tokenizer/output_dir differ between calls.
    """
    # A batch groups several examples together so the model trains on many at
    # once instead of one at a time (much faster). But examples have different
    # lengths (some C#/Java snippets are longer than others), so they can't
    # just be stacked as-is. The data collator pads every example in a batch
    # up to the same length right before it's fed to the model, using values
    # the model treats as "ignore this".
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Seq2SeqTrainingArguments is just a settings object: it doesn't train
    # anything by itself, it only records the choices the Trainer will follow.
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,                 # where checkpoints and logs get saved
        eval_strategy="epoch",                 # run evaluation on the validation set once per epoch
        save_strategy="epoch",                 # save a checkpoint once per epoch (so eval_strategy and save_strategy line up)
        learning_rate=5e-5,                    # how big a step the model takes when correcting its mistakes
        per_device_train_batch_size=16,        # how many examples are grouped in one training batch
        num_train_epochs=10,                   # how many full passes over the training set
        load_best_model_at_end=True,           # after training, automatically reload the checkpoint with the best validation score
        report_to="none",                      # don't send logs to wandb/other external trackers
    )

    # The Trainer is the object that actually runs the training loop: it feeds
    # batches from train_dataset to the model, compares its output to "labels",
    # adjusts the model's weights, and periodically checks eval_dataset -- all
    # using the settings above. Nothing trains yet -- this only configures
    # *how* training should happen; it starts when trainer.train() is called.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    return trainer, training_args


# --- Setting up training for both models ---
trainer_small, training_args_small = build_trainer(model_small, tokenizer_small, tokenized_small, f"{DRIVE_DIR}/results_codet5_small")
trainer_base, training_args_base = build_trainer(model_base, tokenizer_base, tokenized_base, f"{DRIVE_DIR}/results_codet5_base")

# --- Actually training both models ---
# trainer.train() runs the training loop configured above. Thanks to
# load_best_model_at_end=True, once it finishes, trainer.model holds the
# checkpoint that scored best on the validation set, not just the last one.
print("Training codet5-small...")
trainer_small.train()

print("Training codet5-base...")
trainer_base.train()

# --- Saving the fine-tuned models ---
# Saves the best checkpoint (see load_best_model_at_end above) plus its
# tokenizer to disk, so Phase C can reload them without repeating training.
trainer_small.save_model(f"{DRIVE_DIR}/fine_tuned_codet5_small")
tokenizer_small.save_pretrained(f"{DRIVE_DIR}/fine_tuned_codet5_small")

trainer_base.save_model(f"{DRIVE_DIR}/fine_tuned_codet5_base")
tokenizer_base.save_pretrained(f"{DRIVE_DIR}/fine_tuned_codet5_base")

# --- Recording the training parameters used ---
# Phase C/D will compare the two models, so the settings used to train them
# need to be written down somewhere. Kept identical on purpose (see above).
print("Training parameters used (identical for both models):")
print(f"  learning_rate: {training_args_small.learning_rate}")
print(f"  per_device_train_batch_size: {training_args_small.per_device_train_batch_size}")
print(f"  num_train_epochs: {training_args_small.num_train_epochs}")
print("  max_input_length / max_target_length: 256 (preprocess_function default)")
