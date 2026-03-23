"""
Handler for LLM-JP-4 model using OpenAI Harmony response format.

This handler implements prompt formatting and response parsing according to
the OpenAI Harmony template format as specified in:
https://github.com/openai/harmony
"""

import json
from typing import Any, Dict, List, Optional

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override

try:
    from openai_harmony import (
        Conversation,
        DeveloperContent,
        HarmonyEncodingName,
        Message,
        Role,
        SystemContent,
        load_harmony_encoding,
    )

    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False


class LLMjp4Handler(OSSHandler):
    """
    Handler for LLM-JP-4 models using OpenAI Harmony response format.

    The Harmony format enables structured conversation with:
    - Multiple channels (analysis, commentary, final)
    - Tool/function calling support
    - Reasoning output
    - Clear instruction hierarchy
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

        # Initialize Harmony encoding if available
        if HARMONY_AVAILABLE:
            self.harmony_enc = load_harmony_encoding(
                HarmonyEncodingName.HARMONY_GPT_OSS
            )
        else:
            self.harmony_enc = None
            print(
                "Warning: openai-harmony package not installed. "
                "Install with: pip install openai-harmony"
            )

    @override
    def _format_prompt(self, messages: List[Dict], function: List[Dict]) -> str:
        """
        Format messages and functions into Harmony format.

        The Harmony format structure:
        <|start|>system<|message|>System content...<|end|>
        <|start|>developer<|message|>Developer instructions and tools...<|end|>
        <|start|>user<|message|>User query...<|end|>
        <|start|>assistant

        Args:
            messages: List of message dicts with 'role' and 'content'
            function: List of function definitions

        Returns:
            Formatted prompt string ready for model inference
        """
        if self.harmony_enc is not None:
            return self._format_with_harmony_library(messages, function)
        else:
            return self._format_with_manual_template(messages, function)

    def _format_with_harmony_library(
        self, messages: List[Dict], function: List[Dict]
    ) -> str:
        """
        Format using the openai-harmony library.
        """
        harmony_messages = []

        # Add system message
        system_content = SystemContent.new()
        system_content = system_content.with_reasoning_effort("high")
        system_content = system_content.with_required_channels(
            ["analysis", "commentary", "final"]
        )

        # Add function tools if available
        if function:
            # Convert BFCL function format to Harmony tool format
            tool_namespace = self._convert_functions_to_harmony_tools(function)
            if tool_namespace:
                system_content.tools = {"functions": tool_namespace}

        harmony_messages.append(
            Message.from_role_and_content(Role.SYSTEM, system_content)
        )

        # Add developer instructions if we have functions
        if function:
            dev_content = DeveloperContent.new()
            dev_content = dev_content.with_instructions(
                "You are a function calling assistant. "
                "Use the provided functions to help answer user queries. "
                "Always respond with valid JSON for function calls."
            )
            harmony_messages.append(
                Message.from_role_and_content(Role.DEVELOPER, dev_content)
            )

        # Add conversation messages
        for msg in messages:
            role = self._convert_role(msg["role"])
            if role:
                harmony_messages.append(
                    Message.from_role_and_content(role, str(msg["content"]))
                )

        # Create conversation and render
        convo = Conversation.from_messages(harmony_messages)
        try:
            tokens = self.harmony_enc.render_conversation_for_completion(
                convo, Role.ASSISTANT
            )
            return self.harmony_enc.decode_utf8(tokens)
        except Exception as e:
            print(f"Error rendering with Harmony: {e}")
            return self._format_with_manual_template(messages, function)

    def _format_with_manual_template(
        self, messages: List[Dict], function: List[Dict]
    ) -> str:
        """
        Fallback: manually construct Harmony format without the library.
        """
        formatted_prompt = ""

        # System message
        formatted_prompt += "<|start|>system<|message|>"
        formatted_prompt += (
            "You are ChatGPT, a large language model trained by OpenAI.\n"
        )
        formatted_prompt += "Knowledge cutoff: 2024-06\n"
        formatted_prompt += f"Current date: {self._get_current_date()}\n\n"
        formatted_prompt += "Reasoning: high\n\n"
        formatted_prompt += "# Valid channels: analysis, commentary, final. "
        formatted_prompt += "Channel must be included for every message.\n"
        if function:
            formatted_prompt += (
                "Calls to these tools must go to the commentary channel: 'functions'."
            )

        formatted_prompt += "<|end|>\n\n"

        # Developer message with function definitions
        if function:
            formatted_prompt += "<|start|>developer<|message|>"
            formatted_prompt += "# Instructions\n\n"
            formatted_prompt += "You are a function calling assistant. "
            formatted_prompt += (
                "Use the provided functions to help answer user queries.\n\n"
            )
            formatted_prompt += "# Tools\n\n"
            formatted_prompt += "## functions\n\n"
            formatted_prompt += "namespace functions {\n\n"

            for func in function:
                formatted_prompt += self._format_function_typescript(func)
                formatted_prompt += "\n"

            formatted_prompt += "} // namespace functions"
            formatted_prompt += "<|end|>\n"

        # Add conversation messages
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted_prompt += f"<|start|>{role}<|message|>{content}<|end|>"

        # Start assistant response
        formatted_prompt += "<|start|>assistant"

        return formatted_prompt

    def _format_function_typescript(self, func: Dict) -> str:
        """
        Convert function definition to TypeScript-style format for Harmony.

        Example output:
        // Gets the current weather in a location.
        type get_weather = (_: {
          location: string,
          unit?: "celsius" | "fahrenheit", // default: celsius
        }) => any;
        """
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})

        result = f"// {description}\n" if description else ""
        result += f"type {name} = ("

        if parameters and "properties" in parameters:
            result += "_: {\n"
            props = parameters["properties"]
            required = parameters.get("required", [])

            for i, (param_name, param_info) in enumerate(props.items()):
                # Add comment for parameter description
                param_desc = param_info.get("description", "")

                # Determine if optional
                optional_mark = "" if param_name in required else "?"

                # Get type
                param_type = self._convert_json_type_to_ts(param_info)

                result += f"  {param_name}{optional_mark}: {param_type},"

                if param_desc:
                    result += f" // {param_desc}"

                result += "\n"

            result += "}"

        result += ") => any;"

        return result

    def _convert_json_type_to_ts(self, param_info: Dict) -> str:
        """Convert JSON schema type to TypeScript type."""
        json_type = param_info.get("type", "any")

        # Handle enums
        if "enum" in param_info:
            enum_values = param_info["enum"]
            return " | ".join([f'"{v}"' for v in enum_values])

        # Handle basic types
        type_mapping = {
            "string": "string",
            "number": "number",
            "integer": "number",
            "boolean": "boolean",
            "array": "any[]",
            "object": "any",
        }

        return type_mapping.get(json_type, "any")

    def _convert_functions_to_harmony_tools(
        self, functions: List[Dict]
    ) -> Optional[Any]:
        """
        Convert BFCL function format to Harmony ToolNamespaceConfig.
        Returns None if harmony library is not available.
        """
        if not HARMONY_AVAILABLE:
            return None

        from openai_harmony import ToolDescription, ToolNamespaceConfig

        tool_descriptions = []
        for func in functions:
            tool_desc = ToolDescription(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=func.get("parameters"),
            )
            tool_descriptions.append(tool_desc)

        return ToolNamespaceConfig(
            name="functions",
            description="Available function calls",
            tools=tool_descriptions,
        )

    def _convert_role(self, role: str) -> Optional[Any]:
        """Convert string role to Harmony Role enum."""
        if not HARMONY_AVAILABLE:
            return None

        role_mapping = {
            "system": Role.SYSTEM,
            "user": Role.USER,
            "assistant": Role.ASSISTANT,
            "developer": Role.DEVELOPER,
            "tool": Role.TOOL,
        }

        return role_mapping.get(role.lower())

    def _get_current_date(self) -> str:
        """Get current date in format expected by Harmony."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d")

    @override
    def decode_ast(
        self, result: str, language: str, has_tool_call_tag: bool
    ) -> List[Dict]:
        """
        Parse model response to extract function calls.

        Expected format from model:
        [{"name": "func_name", "arguments": {"arg1": "val1", ...}}]

        Returns:
        [{func_name: {arg1: val1, ...}}, ...]
        """
        try:
            # If using Harmony, try to parse with the library first
            if self.harmony_enc is not None:
                parsed = self._parse_with_harmony(result)
                if parsed:
                    return parsed

            # Fallback to manual parsing
            return self._parse_function_calls_manual(result)

        except Exception as e:
            print(f"Error parsing function calls: {e}")
            return []

    def _parse_with_harmony(self, result: str) -> Optional[List[Dict]]:
        """
        Parse result using Harmony library.
        """
        try:
            # Tokenize the result
            # Note: This is a simplified approach. In production, you'd want
            # to handle the full token stream from the model.
            messages = self.harmony_enc.parse_messages_from_completion_tokens(
                result,
                role=Role.ASSISTANT,
                strict=False,  # Allow permissive parsing
            )

            # Extract function calls from parsed messages
            function_calls = []
            for msg in messages:
                # Look for JSON function call format in message content
                content = str(msg.content) if hasattr(msg, "content") else str(msg)
                calls = self._extract_function_calls_from_content(content)
                function_calls.extend(calls)

            return function_calls if function_calls else None

        except Exception as e:
            print(f"Harmony parsing failed: {e}")
            return None

    def _parse_function_calls_manual(self, result: str) -> List[Dict]:
        """
        Manually parse function calls from result string.
        """
        # Clean up the result - remove markdown code blocks if present
        result = result.strip()
        if result.startswith("```"):
            # Remove markdown code block markers
            lines = result.split("\n")
            result = "\n".join(lines[1:-1] if len(lines) > 2 else lines)

        result = result.strip()

        # Try to parse as JSON
        try:
            function_calls = json.loads(result)
        except json.JSONDecodeError:
            # Try to find JSON in the string
            import re

            json_match = re.search(r"\[.*\]", result, re.DOTALL)
            if json_match:
                function_calls = json.loads(json_match.group(0))
            else:
                return []

        # Ensure it's a list
        if isinstance(function_calls, dict):
            function_calls = [function_calls]

        # Convert to expected format: [{func_name: {params}}]
        execution_list = []
        for func_call in function_calls:
            if isinstance(func_call, dict) and "name" in func_call:
                name = func_call["name"]
                params = func_call.get("arguments", {})
                execution_list.append({name: params})

        return execution_list

    def _extract_function_calls_from_content(self, content: str) -> List[Dict]:
        """Extract function calls from message content."""
        try:
            # Look for JSON array or object
            import re

            json_match = re.search(r"\[.*?\]|\{.*?\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    data = [data]

                result = []
                for item in data:
                    if "name" in item and "arguments" in item:
                        result.append({item["name"]: item["arguments"]})
                return result
        except:
            pass

        return []

    @override
    def decode_execute(self, result: str, has_tool_call_tag: bool) -> List[str]:
        """
        Parse model response for execution.

        Returns:
        ["func_name(arg1='val1', arg2='val2')", ...]
        """
        try:
            # Parse function calls
            parsed_calls = self._parse_function_calls_manual(result)

            # Convert to execution strings
            execution_list = []
            for call_dict in parsed_calls:
                for func_name, params in call_dict.items():
                    param_str = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
                    execution_list.append(f"{func_name}({param_str})")

            return execution_list

        except Exception as e:
            print(f"Error in decode_execute: {e}")
            return []

    @override
    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """
        Add execution results back to the conversation for multi-turn interaction.
        """
        for execution_result in execution_results:
            # Use 'tool' role for execution results in Harmony format
            inference_data["message"].append(
                {
                    "role": "tool",
                    "content": execution_result,
                }
            )

        return inference_data
