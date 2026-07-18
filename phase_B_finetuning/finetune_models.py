"""
Phase B -- Model fine-tuning.

Loads pretrained tokenizers and models (Salesforce/codet5-small,
Salesforce/codet5-base, and vanilla t5-base as a non-code-pretrained
baseline, referred to as t5vanilla throughout), tokenizes the dataset
prepared in Phase A with preprocess_function, and fine-tunes all three
models on the training set, using the validation set to select the
best checkpoint.
Records the training parameters used (learning rate, batch size, number
of epochs, maximum length).
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from phase_A_preparation.prepare_dataset import load_and_prepare_dataset, preprocess_function

# Saved to Google Drive instead of local disk so nothing is lost if the
# Colab session disconnects. Assumes drive.mount("/content/drive") already ran.
DRIVE_DIR = "/content/drive/MyDrive/csharp_to_java_project"

# codet5-small
tokenizer_small = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
model_small = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

# codet5-base
tokenizer_base = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model_base = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")

# t5vanilla: same size as codet5-base (t5-base, ~220M params) but pretrained
# only on generic text (C4), not on code. Kept at the same size as
# codet5-base on purpose, so the only variable that changes between the two
# is the pretraining corpus, not the parameter count.
tokenizer_t5vanilla = AutoTokenizer.from_pretrained("t5-base")
model_t5vanilla = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

prepared_dataset = load_and_prepare_dataset()

tokenized_small = prepared_dataset.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer_small}, batched=True)
tokenized_base = prepared_dataset.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer_base}, batched=True)
tokenized_t5vanilla = prepared_dataset.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer_t5vanilla}, batched=True)


def build_trainer(model, tokenizer, tokenized_dataset, output_dir):
    """
    Builds a Seq2SeqTrainer ready to fine-tune 'model' on 'tokenized_dataset'.

    Same settings used for every model passed in, so configurations stay
    comparable across models -- only model/tokenizer/output_dir differ.
    """
    # Pads every example in a batch to the same length before it's fed to
    # the model, since C#/Java snippets vary in length and can't otherwise
    # be stacked into a single tensor.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Settings object: records the choices the Trainer will follow, doesn't train anything by itself.
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,                 # where checkpoints and logs get saved
        eval_strategy="epoch",                 # run evaluation on the validation set once per epoch
        save_strategy="epoch",                 # save a checkpoint once per epoch
        learning_rate=5e-5,
        per_device_train_batch_size=8,          # kept low enough to fit codet5-base on a 16GB GPU
        num_train_epochs=10,
        load_best_model_at_end=True,           # reload the checkpoint with the best validation score after training
        report_to="none",                      # don't send logs to wandb/other external trackers
    )

    # Feeds batches from train_dataset to the model, compares output to
    # "labels", adjusts weights, and periodically checks eval_dataset.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    return trainer, training_args


trainer_small, training_args_small = build_trainer(model_small, tokenizer_small, tokenized_small, f"{DRIVE_DIR}/results_codet5_small")
trainer_base, training_args_base = build_trainer(model_base, tokenizer_base, tokenized_base, f"{DRIVE_DIR}/results_codet5_base")
trainer_t5vanilla, training_args_t5vanilla = build_trainer(model_t5vanilla, tokenizer_t5vanilla, tokenized_t5vanilla, f"{DRIVE_DIR}/results_t5vanilla")

# Each model is trained, saved, and cleared from GPU memory before the next
# one starts -- otherwise codet5-small's memory stays reserved and
# codet5-base (much bigger) runs out of room on top of it.
print("Training codet5-small...")
trainer_small.train()
trainer_small.save_model(f"{DRIVE_DIR}/fine_tuned_codet5_small")
tokenizer_small.save_pretrained(f"{DRIVE_DIR}/fine_tuned_codet5_small")

del trainer_small, model_small
torch.cuda.empty_cache()

print("Training codet5-base...")
trainer_base.train()
trainer_base.save_model(f"{DRIVE_DIR}/fine_tuned_codet5_base")
tokenizer_base.save_pretrained(f"{DRIVE_DIR}/fine_tuned_codet5_base")

del trainer_base, model_base
torch.cuda.empty_cache()

print("Training t5vanilla...")
trainer_t5vanilla.train()
trainer_t5vanilla.save_model(f"{DRIVE_DIR}/fine_tuned_t5vanilla")
tokenizer_t5vanilla.save_pretrained(f"{DRIVE_DIR}/fine_tuned_t5vanilla")

del trainer_t5vanilla, model_t5vanilla
torch.cuda.empty_cache()

print("Training parameters used (identical for all three models):")
print(f"  learning_rate: {training_args_small.learning_rate}")
print(f"  per_device_train_batch_size: {training_args_small.per_device_train_batch_size}")
print(f"  num_train_epochs: {training_args_small.num_train_epochs}")
print("  max_input_length / max_target_length: 256 (preprocess_function default)")
