from overrides import override

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler

TASK_INSTRUCTION = """
Based on the previous context and API request history, generate an API request or a response as an AI assistant.""".strip()
FORMAT_INSTRUCTION = """
The output should be of the JSON format, which specifies a list of generated function calls. The example format is as follows, please make sure the parameter type is correct. If no function call is needed, please make tool_calls an empty list "[]".
```
{"thought": "the thought process, or an empty string", "tool_calls": [{"name": "api_name1", "arguments": {"argument1": "value1", "argument2": "value2"}}]}
```
""".strip()


class LLMjp3Handler(OSSHandler):
    """
    This the handler for the LLM-jp-3 models in function calling mode.
    According to the Llama model card, function calling should be handled differently
    than what is suggested by the standard Hugging Face chat template.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name.replace("-FC", "")

    @override
    def _format_prompt(self, messages, function):
        # For Llama 4 series, they use a different set of tokens than Llama 3
        formatted_prompt = TASK_INSTRUCTION + "\n" + FORMAT_INSTRUCTION
        for message in messages:
            if message["role"] == "user":
                formatted_prompt += "user input:" + str(message["content"]) + "\n"
            elif message["role"] == "assistant":
                formatted_prompt += "assistant: " + str(message["content"]) + "\n"
        return formatted_prompt

    @override
    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        for execution_result in execution_results:
            # Llama uses the `ipython` role for execution results
            inference_data["message"].append(
                {
                    "role": "ipython",
                    "content": execution_result,
                }
            )

        return inference_data

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        """
        [{func1: {param1: val1, param2: val2, ...}}, {func2: {param1: val1, param2: val2, ...}}, ...]
        の形式にしないといけない
        """
        function_calls = eval(result)
        if isinstance(function_calls, dict):
            function_calls = [function_calls]

        execution_list = []
        for func_call in function_calls:
            name = func_call["name"]
            params = func_call["arguments"]
            # execution_list.append(
            #     f"{name}({','.join([f'{k}={repr(v)}' for k, v in params.items()])})"
            # )
            execution_list.append(
                {
                    name: params,
                }
            )
        # print(f"function calls: {function_calls}")
        # print(f"converted function calls: {execution_list}")
        return execution_list

    @override
    def decode_execute(self, result, has_tool_call_tag):
        function_calls = eval(result)
        if isinstance(function_calls, dict):
            function_calls = [function_calls]

        execution_list = []
        for func_call in function_calls:
            name = func_call["name"]
            params = func_call["arguments"]
            execution_list.append(
                f"{name}({','.join([f'{k}={repr(v)}' for k, v in params.items()])})"
            )
        # print(f"function calls: {function_calls}")
        # print(f"converted function calls: {execution_list}")
        return execution_list
