"""This is a file containing the prompter for a llama model finetuned for
function calling by me. You can find the model this was made for at
https://huggingface.co/rizerphe/CodeLlama-function-calling-6320-7b-Instruct-GGUF
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..prompter import FunctionType


class CodeLlamaFunctionCallingPrompter:
    """A prompter for code llama function calling models"""

    def prompt(
        self,
        prompt: str,
        functions: list[FunctionType],
        function_to_call: str | None = None,
    ) -> list[bytes | int]:
        """Generate the llama prompt

        Args:
            prompt (str): The prompt to generate the response to
            functions (list[FunctionType]): The functions to generate the response from
            function_to_call (str | None): The function to call. Defaults to None.

        Returns:
            list[bytes | int]: The llama prompt, a function selection prompt if no
                function is specified, or a function argument prompt if a function is
                specified
        """
        functions_summary = "\n".join(
            f"<function>{json.dumps(f, indent=4)}" for f in functions
        )
        f_start = (function_to_call + "\n") if function_to_call else ""
        return [
            1,
            f"[INST] <<SYS>>\n<function>Available functions:\n{functions_summary}"
            f"\n<<SYS>>\n\n{prompt} [/INST]<function>{f_start}".encode("utf-8"),
        ]
