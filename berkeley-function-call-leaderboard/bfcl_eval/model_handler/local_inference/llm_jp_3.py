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
        formatted_prompt = ""
        for message in messages:
            if messages["role"] == "user":
                formatted_prompt += "user input:" + message["content"] + "\n"
            elif messages["role"] == "assistant":
                formatted_prompt += "assistant: " + message["content"] + "\n"
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
