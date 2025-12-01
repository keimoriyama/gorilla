from overrides import override

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler


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
        """
        [{'role': 'system', 'content': 'You are an expert in composing functions.You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\n\nYou should only return the function calls in your response.\n\nIf you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  You SHOULD NOT include any other text in the response.\n\nAt each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user\'s request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.\n\nHere is a list of functions in json format that you can invoke.\n[\n    {\n        "name": "answer_question",\n        "description": "This function transfers the chat interaction to a human agent when the automated system encounters a question that it cannot answer. Note that the provided function is in Python 3 syntax.",\n        "parameters": {\n            "type": "dict",\n            "required": [\n                "statement"\n            ],\n            "properties": {\n                "statement": {\n                    "type": "string",\n                    "description": "The question posed by the user that needs to be transferred to a human agent."\n                },\n                "urgency": {\n                    "type": "string",\n                    "description": "The level of urgency for the question to be answered.",\n                    "enum": [\n                        "low",\n                        "medium",\n                        "high"\n                    ],\n                    "default": "medium"\n                },\n                "language": {\n                    "type": "string",\n                    "description": "The language in which the question is asked, using ISO 639-1 codes (e.g., \'en\' for English, \'es\' for Spanish).",\n                    "default": "en"\n                }\n            }\n        }\n    }\n]\n\n\ncall HANDOVER function to transfer the request if user asks a question.'}, {'role': 'user', 'content': 'Can you tell me what is the minimum package arrival time? '}]
        """
        formatted_prompt = messages[0]["content"] + "\n"
        formatted_prompt += "User query: " + messages[1]["content"] + "\n"
        # print(formatted_prompt)
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
