# inference.py

import argparse
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Transformer model.")
    parser.add_argument("--model_path", type=str, default="./output",
                        help="Path to the fine-tuned model directory.")
    parser.add_argument("--text", type=str, required=True,
                        help="Input text for classification.")
    args = parser.parse_args()

    # Create a pipeline for text classification
    clf_pipe = pipeline("text-classification", model=args.model_path, tokenizer=args.model_path)

    # Run inference on the provided text
    output = clf_pipe(args.text)
    print(f"Input: {args.text}")
    print(f"Output: {output}")

if __name__ == "__main__":
    main()
