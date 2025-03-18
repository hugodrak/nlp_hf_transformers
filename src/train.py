# train.py

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

from data_loader import load_and_prepare_dataset
from model import load_model_and_tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

def main():
    parser = argparse.ArgumentParser(description="Train a Hugging Face Transformer for classification.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Model name or path on Hugging Face Hub.")
    parser.add_argument("--dataset_name", type=str, default="imdb",
                        help="Dataset name or path recognized by 'datasets'.")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="Number of labels for classification task.")
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Evaluation batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max input sequence length.")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model checkpoints.")
    args = parser.parse_args()

    # 1. Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        num_labels=args.num_labels
    )

    # 2. Load & tokenize data
    # For the 'imdb' dataset, there's a "train" and "test" split by default.
    # We'll treat "test" as our validation in this example. Real usage might differ.
    dataset_splits = {
        "train": "train",
        "validation": "test"
    }
    data = load_and_prepare_dataset(
        args.dataset_name,
        tokenizer,
        args.max_length,
        split_mapping=dataset_splits
    )

    train_dataset = data["train"]
    eval_dataset = data["validation"]

    # 3. Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # or "eval_loss", "f1", etc.
    )

    # 4. Create data collator (for dynamic padding)
    data_collator = DataCollatorWithPadding(tokenizer)

    # 5. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 6. Train & Evaluate
    trainer.train()
    trainer.evaluate()

    # 7. Save final model
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
