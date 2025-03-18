# model.py

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase
)

def load_model_and_tokenizer(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2  # Binary classification by default
) -> (PreTrainedModel, PreTrainedTokenizerBase):
    """
    Loads a pre-trained model and tokenizer for sequence classification.

    Args:
        model_name: A model name from Hugging Face Hub (e.g. "bert-base-uncased").
        num_labels: Number of classes for classification.

    Returns:
        model: An AutoModelForSequenceClassification instance.
        tokenizer: An AutoTokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    return model, tokenizer
