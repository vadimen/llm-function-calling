"""This is a file containing the prompter for a llama model finetuned for
function calling by me. You can find the model this was made for at
https://huggingface.co/rizerphe/CodeLlama-function-calling-6320-7b-Instruct-GGUF
"""
from __future__ import annotations
import json
from typing import Literal, NotRequired, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from ..prompter import FunctionType, ShouldCallResponse


class FunctionCall(TypedDict):
    name: str
    arguments: str


class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "function"]
    name: NotRequired[str]
    content: NotRequired[str]
    function_call: NotRequired[FunctionCall]


class CodeLlamaFunctionCallingPrompter:
    """A prompter for code llama function calling models"""

    def _user_message(self, message: ChatMessage) -> str:
        if message["role"] == "user":
            return f"[INST] {message.get('content', '')} [/INST]"
        return f"[INST] <function>{message.get('content', '')} [/INST]"

    def _assistant_message(self, message: ChatMessage) -> str:
        if "content" in message:
            return f" {message['content']}"
        if "function_call" not in message:
            return ""
        return (
            "<function>"
            + message["function_call"]["name"]
            + "\n"
            + message["function_call"]["arguments"]
        )

    def _chat_prompt(
        self,
        chat: list[ChatMessage],
        functions: list[FunctionType],
        function_to_call: str | None = None,
        use_function: bool = False,
    ) -> list[bytes | int]:
        functions_summary = "\n".join(
            f"<function>{json.dumps(f, indent=4)}" for f in functions
        )
        system_prompt = (
            f"<<SYS>>\n<function>Available functions:\n{functions_summary} <</SYS>>\n\n"
        )
        user_message: ChatMessage | None = None
        result: list[int | bytes] = [1]
        for i, message in enumerate(chat):
            if i == 0:
                if message["role"] != "user":
                    raise ValueError("First message must be from user")
                if "content" not in message:
                    raise ValueError("First message must have content")
                message["content"] = system_prompt + message["content"]
            if message["role"] in ["user", "function"]:
                if user_message is not None:
                    result.append(self._user_message(user_message).encode("utf-8"))
                    result.append(2)
                    result.append(1)
                user_message = message
            if message["role"] == "assistant":
                if user_message is None:
                    ...
                else:
                    result.append(
                        (
                            self._user_message(user_message)
                            + self._assistant_message(message)
                        ).encode("utf-8")
                    )
                    result.append(2)
                    result.append(1)
                    user_message = None
        if user_message is not None:
            result.append(self._user_message(user_message).encode("utf-8"))
            result.append(2)
            result.append(1)

        f_start = (
            ("<function>" + ((function_to_call + "\n") if function_to_call else ""))
            if use_function
            else ""
        )
        if result and result[-1] == 1:
            result.pop()
            result.pop()
        if not result or isinstance(result[-1], int):
            result.append(f_start.encode("utf-8"))
        else:
            result[-1] += f_start.encode("utf-8")

        return result

    def prompt(
        self,
        prompt: str | list[ChatMessage],
        functions: list[FunctionType],
        function_to_call: str | None = None,
    ) -> list[bytes | int]:
        """Generate the llama prompt

        Args:
            prompt (str | list[ChatMessage]): The prompt to generate the response from
            functions (list[FunctionType]): The functions to generate the response from
            function_to_call (str | None): The function to call. Defaults to None.

        Returns:
            list[bytes | int]: The llama prompt, a function selection prompt if no
                function is specified, or a function argument prompt if a function is
                specified
        """
        return (
            self._chat_prompt(prompt, functions, function_to_call, True)
            if isinstance(prompt, list)
            else self._chat_prompt(
                [{"role": "user", "content": prompt}], functions, function_to_call, True
            )
        )

    def should_call_prompt(
        self, prompt: str | list[ChatMessage], functions: list[FunctionType]
    ) -> tuple[list[bytes | int], ShouldCallResponse]:
        """Check if a function should be called

        Args:
            prompt (str | list[ChatMessage]): The prompt to generate the response from
            functions (list[FunctionType]): The functions to choose from

        Returns:
            tuple[str, ShouldCallResponse]: The function to call and the response
        """
        return (
            (
                self._chat_prompt(prompt, functions, None, False)
                if isinstance(prompt, list)
                else self._chat_prompt(
                    [{"role": "user", "content": prompt}], functions, None, False
                )
            ),
            {"if_should_call": ["<function>"], "if_not_should_call": [" "]},
        )

    def natural_language_prompt(
        self, prompt: str, functions: list[FunctionType]
    ) -> list[bytes | int]:
        """Prompt the model to generate a natural language response

        Args:
            prompt (str): The natural language part of the prompt
            functions (list[FunctionType]): The functions to choose from

        Returns:
            list[bytes | int]: The natural language prompt
        """
        return self.should_call_prompt(prompt, functions)[0]
