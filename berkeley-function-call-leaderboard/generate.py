import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # プロンプトの構築
    with open("./bfcl_eval/data/BFCL_v4_simple_python.json", "r") as f:
        data = json.load(f)

    # モデルの応答


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using a pre-trained language model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Name of the pre-trained model to use.",
    )
    args = parser.parse_args()
    main(args)
