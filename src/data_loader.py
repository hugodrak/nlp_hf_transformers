# data_loader.py

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from typing import Dict, Any

def load_and_prepare_dataset(
    dataset_name_or_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    split_mapping: Dict[str, str] = None
):
    """
    Loads a dataset from the Hugging Face 'datasets' library or local files,
    and tokenizes it for a classification task.

    Args:
        dataset_name_or_path: The name of a dataset on Hugging Face Hub (e.g., "imdb"),
                              or a path to local data (e.g., "data/train.csv").
        tokenizer: A Hugging Face tokenizer (e.g., BertTokenizer).
        max_length: The maximum sequence length for tokenization.
        split_mapping: An optional dictionary mapping, e.g., {"train": "train", "validation": "test"}
                       if you need to rename splits.
    Returns:
        A dictionary containing "train", "validation", and optionally "test" dataset splits,
        each ready for training with tokenized inputs.
    """
    if split_mapping is None:
        # Default splits if not provided
        split_mapping = {"train": "train", "validation": "validation"}

    # Attempt to load a dataset by name or path
    dataset = load_dataset(dataset_name_or_path)

    # Some datasets might only have train/test splits, or different names.
    # We’ll map them to "train"/"validation" here:
    needed_splits = list(split_mapping.values())
    for s in needed_splits:
        if s not in dataset:
            raise ValueError(f"Split '{s}' not found in dataset. Available splits: {dataset.keys()}")

    def tokenize_function(examples: Dict[str, Any]):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # Tokenize the dataset
    tokenized_dataset = {}
    for split_name, actual_name in split_mapping.items():
        tokenized_dataset[split_name] = dataset[actual_name].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]  # keep 'label' in dataset
        )

    return tokenized_dataset
