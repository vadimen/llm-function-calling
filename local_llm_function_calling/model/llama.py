"""A container for llama-cpp models"""
from __future__ import annotations
import json
from typing import Generator, Iterator, TYPE_CHECKING

from llama_cpp import Llama, LlamaGrammar, sys

if TYPE_CHECKING:
    from ..prompter import FunctionType


class LogitLlama(Llama):
    """A wrapper for llama-cpp models that returns logits instead of tokens"""

    def generate_logits(
        self,
        tokens: list[int],
        reset: bool = True,
        grammar: LlamaGrammar | None = None,
    ) -> Generator[list[float], list[int], None]:
        """Create a generator of tokens from a prompt.

        Args:
            tokens: The prompt tokens.
            top_k: The top-k sampling parameter.
            top_p: The top-p sampling parameter.
            temp: The temperature parameter.
            repeat_penalty: The repeat penalty parameter.
            reset: Whether to reset the model state.

        Yields:
            The token logits, expecting the next token(s) in return.
        """
        assert self.ctx is not None
        if reset and len(self._input_ids) > 0:
            longest_prefix = 0
            for a, b in zip(self._input_ids, tokens[:-1]):
                if a == b:
                    longest_prefix += 1
                else:
                    break
            if longest_prefix > 0:
                if self.verbose:
                    print("Llama.generate: prefix-match hit", file=sys.stderr)
                reset = False
                tokens = tokens[longest_prefix:]
                self.n_tokens = longest_prefix

        if reset:
            self.reset()

        if grammar is not None:
            grammar.reset()

        while True:
            self.eval(tokens)
            tokens = yield self.eval_logits[0]
            if tokens is None:
                break


class CodeLlamaFc:
    """A prompter for code llama function calling models"""

    def function_descriptions(
        self, functions: list[FunctionType], function_to_call: str
    ) -> list[str]:
        """Get the descriptions of the functions

        Args:
            functions (list[FunctionType]): The functions to get the descriptions of
            function_to_call (str): The function to call

        Returns:
            list[str]: The descriptions of the functions
                (empty if the function doesn't exist or has no description)
        """
        return [
            "Function description: " + function["description"]
            for function in functions
            if function["name"] == function_to_call and "description" in function
        ]

    def function_parameters(
        self, functions: list[FunctionType], function_to_call: str
    ) -> str:
        """Get the parameters of the function

        Args:
            functions (list[FunctionType]): The functions to get the parameters of
            function_to_call (str): The function to call

        Returns:
            str: The parameters of the function as a JSON schema
        """
        return next(
            json.dumps(function["parameters"]["properties"], indent=4)
            for function in functions
            if function["name"] == function_to_call
        )

    def function_data(
        self, functions: list[FunctionType], function_to_call: str
    ) -> str:
        """Get the data for the function

        Args:
            functions (list[FunctionType]): The functions to get the data for
            function_to_call (str): The function to call

        Returns:
            str: The data necessary to generate the arguments for the function
        """
        return "\n".join(
            self.function_descriptions(functions, function_to_call)
            + [
                "Function parameters should follow this schema:",
                "```jsonschema",
                self.function_parameters(functions, function_to_call),
                "```",
            ]
        )

    def function_summary(self, function: FunctionType) -> str:
        """Get a summary of a function

        Args:
            function (FunctionType): The function to get the summary of

        Returns:
            str: The summary of the function, as a bullet point
        """
        return f"- {function['name']}" + (
            f" - {function['description']}" if "description" in function else ""
        )

    def functions_summary(self, functions: list[FunctionType]) -> str:
        """Get a summary of the functions

        Args:
            functions (list[FunctionType]): The functions to get the summary of

        Returns:
            str: The summary of the functions, as a bulleted list
        """
        return "Available functions:\n" + "\n".join(
            self.function_summary(function) for function in functions
        )

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
        system = (
            "Help choose the appropriate function "
            "to call to answer the user's question."
            if function_to_call is None
            else f"Define the arguments for {function_to_call} "
            "to answer the user's question."
        )
        data = (
            self.function_data(functions, function_to_call)
            if function_to_call
            else self.functions_summary(functions)
        )
        response_start = (
            f"Here are the arguments for the `{function_to_call}` function: ```json\n"
            if function_to_call
            else "Here's the function the user should call: "
        )
        return [
            1,
            f"[INST] <<SYS>>\n{system}\n\n{data}\n<</SYS>>\n\n{prompt} [/INST]"
            f" {response_start}".encode("utf-8"),
        ]


class LlamaInstructPrompter:
    """A prompter for Llama2 instruct models"""

    def function_descriptions(
        self, functions: list[FunctionType], function_to_call: str
    ) -> list[str]:
        """Get the descriptions of the functions

        Args:
            functions (list[FunctionType]): The functions to get the descriptions of
            function_to_call (str): The function to call

        Returns:
            list[str]: The descriptions of the functions
                (empty if the function doesn't exist or has no description)
        """
        return [
            "Function description: " + function["description"]
            for function in functions
            if function["name"] == function_to_call and "description" in function
        ]

    def function_parameters(
        self, functions: list[FunctionType], function_to_call: str
    ) -> str:
        """Get the parameters of the function

        Args:
            functions (list[FunctionType]): The functions to get the parameters of
            function_to_call (str): The function to call

        Returns:
            str: The parameters of the function as a JSON schema
        """
        return next(
            json.dumps(function["parameters"]["properties"], indent=4)
            for function in functions
            if function["name"] == function_to_call
        )

    def function_data(
        self, functions: list[FunctionType], function_to_call: str
    ) -> str:
        """Get the data for the function

        Args:
            functions (list[FunctionType]): The functions to get the data for
            function_to_call (str): The function to call

        Returns:
            str: The data necessary to generate the arguments for the function
        """
        return "\n".join(
            self.function_descriptions(functions, function_to_call)
            + [
                "Function parameters should follow this schema:",
                "```jsonschema",
                self.function_parameters(functions, function_to_call),
                "```",
            ]
        )

    def function_summary(self, function: FunctionType) -> str:
        """Get a summary of a function

        Args:
            function (FunctionType): The function to get the summary of

        Returns:
            str: The summary of the function, as a bullet point
        """
        return f"- {function['name']}" + (
            f" - {function['description']}" if "description" in function else ""
        )

    def functions_summary(self, functions: list[FunctionType]) -> str:
        """Get a summary of the functions

        Args:
            functions (list[FunctionType]): The functions to get the summary of

        Returns:
            str: The summary of the functions, as a bulleted list
        """
        return "Available functions:\n" + "\n".join(
            self.function_summary(function) for function in functions
        )

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
        system = (
            "Help choose the appropriate function "
            "to call to answer the user's question."
            if function_to_call is None
            else f"Define the arguments for {function_to_call} "
            "to answer the user's question."
        )
        data = (
            self.function_data(functions, function_to_call)
            if function_to_call
            else self.functions_summary(functions)
        )
        response_start = (
            f"Here are the arguments for the `{function_to_call}` function: ```json\n"
            if function_to_call
            else "Here's the function the user should call: "
        )
        return [
            1,
            f"[INST] <<SYS>>\n{system}\n\n{data}\n<</SYS>>\n\n{prompt} [/INST]"
            f" {response_start}".encode("utf-8"),
        ]


class LlamaGeneration:
    """A generation sequence for llama-cpp models"""

    def __init__(self, model: LogitLlama, prefix: list[bytes | int]) -> None:
        """Create a generation sequence

        Args:
            model (LogitLlama): The model to use for generation
            prefix (str): The generation prefix
        """
        self.model = model
        self.generated: list[int] = []
        self.prompt: list[int] = sum(
            (
                [item] if isinstance(item, int) else self.model.tokenize(item, False)
                for item in prefix
            ),
            [],
        )

        self.generation = self.model.generate_logits(self.prompt)
        next(self.generation)

    def get_sorted_tokens(self) -> Iterator[int]:
        """Get the tokens sorted by probability

        Yields:
            The next of the tokens sorted by probability
        """
        probabilities = self.model.eval_logits[0]
        for token_id, _ in sorted(
            enumerate(probabilities), key=lambda item: item[1], reverse=True
        ):
            try:
                if self.get_generated(token_id):
                    yield token_id
            except UnicodeDecodeError:
                continue

    def register_token(self, token: int) -> None:
        """Select the token for this generation step

        Args:
            token (int): The token to select
        """
        self.generated.append(token)
        self.generation.send([token])

    def get_generated(self, candidate: int | None = None) -> str:
        """Get the generated sequence

        Args:
            candidate (int | None): The token to add to the sequence

        Returns:
            str: The generated sequence
        """
        return self.model.detokenize(
            self.generated + ([candidate] if candidate else [])
        ).decode("utf-8")


class LlamaModel:
    """A llama-cpp model"""

    def __init__(
        self,
        model: LogitLlama | str,
    ) -> None:
        """Create a huggingface model

        Args:
            model (LogitLlama | str): The model to use for generation,
                or the path to the model
        """
        if isinstance(model, str):
            self.model = LogitLlama(model)
        else:
            self.model = model

    def start_generation(self, prefix: list[bytes | int]) -> LlamaGeneration:
        """Start a new generation sequence

        Args:
            prefix (list[int]): The generation prefix

        Returns:
            LlamaGeneration: The generation sequence initialized with the prefix
        """
        return LlamaGeneration(self.model, prefix)

    def default_prompter(self) -> LlamaInstructPrompter:
        """Get the default prompter for this model

        Returns:
            LlamaInstructPrompter: The default prompter for this model
        """
        return LlamaInstructPrompter()

    def generate_from_prompt(
        self,
        prefix: list[bytes | int],
        max_tokens: int | None = None,
    ) -> str:
        """Generate a string from a prompt

        Args:
            prefix (list[int]): The prompt to generate a response to
            max_tokens (int | None): The maximum number of tokens to generate

        Returns:
            str: The generated value
        """
        prompt = sum(
            (
                [item] if isinstance(item, int) else self.model.tokenize(item, False)
                for item in prefix
            ),
            [],
        )
        first_token_logits = next(self.model.generate_logits(prompt))
        tokens = [1]
        # This magic is to enforce the first character to be a space
        for token_id, _ in sorted(
            enumerate(first_token_logits), key=lambda item: item[1], reverse=True
        ):
            if (
                self.model.detokenize([token_id])
                .decode("utf-8", errors="ignore")
                .startswith(" ")
            ):
                tokens.append(token_id)
                prompt.append(token_id)
                break
        for i, token in enumerate(self.model.generate(prompt)):
            if token == self.model.token_eos():
                break
            tokens.append(token)
            if max_tokens is not None and i >= max_tokens:
                break
        return self.model.detokenize(tokens).decode("utf-8")
