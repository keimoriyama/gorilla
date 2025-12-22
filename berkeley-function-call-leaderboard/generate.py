import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """
Based on the previous context and API request history, generate an API request or a response as an AI assistant.\n\nThe output should be of the JSON format, which specifies a list of generated function calls. The example format is as follows, please make sure the parameter type is correct. If no function call is needed, please make tool_calls an empty list \"[]\".\n```\n{\"thought\": \"the thought process, or an empty string\", \"tool_calls\": [{\"name\": \"api_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}}]}\n```\n\nThe available APIs are as follows:\n${functions}
""".strip()
USER_ASSISTANT_PROMPT = """
User query: ${user_query}
Assistant:
"""


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # プロンプトの構築
    data = []
    with open("./bfcl_eval/data/BFCL_v4_simple_python.json", "r") as f:
        lines = f.readlines()
        for l in lines:
            data.append(json.loads(l))

    item = data[0]
    # import ipdb;ipdb.set_trace()
    prompt = item["question"][0][0]["content"]
    functions = item["function"]
    system_prompt = SYSTEM_PROMPT.replace("${functions}", json.dumps(functions))
    user_prompt = USER_ASSISTANT_PROMPT.replace("${user_query}", prompt)
    prompt = system_prompt + "\n\n" + user_prompt
    # モデルの応答
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0][len(inputs) :], skip_special_tokens=True)
    print("=== Prompt ===")
    print(prompt)
    print("=== Response ===")
    print(response)


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
