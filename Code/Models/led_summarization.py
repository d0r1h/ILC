import pandas as pd
import os, sys, argparse
from datasets import load_metric, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--text_column", type=str)
    parser.add_argument("--summary_column", type=str)
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--max_output_length", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--output_dir", type=str, required=True)

    args, _ = parser.parse_known_args()

    dataset = load_dataset("d0r1h/ILC")

    train_dataset = dataset["train"].select(range(1000))
    val_dataset = dataset["train"].select(range(300))

    rouge = load_metric("rouge")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name, gradient_checkpointing=True, use_cache=False
    )

    max_input_length = args.max_input_length
    max_output_length = args.max_output_length
    batch_size = args.batch_size

    def process_data_to_model_inputs(batch):

        inputs = tokenizer(
            batch[args.text_column],
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
        )

        outputs = tokenizer(
            batch[args.summary_column],
            padding="max_length",
            truncation=True,
            max_length=max_output_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch

    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=train_dataset.column_names,
    )

    val_dataset = val_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=val_dataset.column_names,
    )

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    model.config.num_beams = args.num_beams
    model.config.max_length = 512
    model.config.min_length = 100
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        output_dir=args.output_dir,
        logging_steps=5,
        eval_steps=10,
        save_steps=10,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
