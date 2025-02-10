
from .deepinfra_helper import DeepInfraHelper
from typing import Optional, Literal, List, Dict

class LLMHelper:
    def __init__(self):
        self._replicate = None
        self._openai = None
        self._hyperbolic = None
        self._deepinfra = None
        self._rapidapi_swiftapi = None

    @property
    def deepinfra(self):
        if not self._deepinfra:
            self._deepinfra = DeepInfraHelper()
        return self._deepinfra

    async def get_completion(
        self, 
        messages: list,
        prompt: str = "",
        provider: Literal["replicate", "openai", "hyperbolic", "deepinfra", "rapidapi_swiftapi"] = "replicate",
        system_prompt: Optional[str] = None,
        llm_name: Optional[str] = None,
        max_new_tokens: int = 5000,
        temperature: float = 0.5,
        top_p: float = 1.0,
        presence_penalty: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Get completion from specified LLM provider
        """
        try:
            if provider == "deepinfra":
                return await self.deepinfra.get_completion(
                    messages=messages,
                    model=llm_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream
                )
                raise ValueError(f"Unknown provider: {provider}")
                
        except Exception as e:
            print(f"Error in LLMHelper.get_completion: {e}")
            raise

    async def generate_with_constraint(
        self,
        model: str,
        messages: List[Dict[str, str]],
        constraint: callable,
        provider: Literal["deepinfra"] = "deepinfra",
        max_attempts: int = 3,
        temperature: float = 0.1
    ) -> str:
        """
        Generate text with constraints from specified provider
        
        Args:
            messages: List of conversation messages
            constraint: Callable that returns (is_valid, is_complete) tuple
            provider: LLM provider (currently only deepinfra supported)
            max_attempts: Maximum number of retries for invalid responses
            temperature: Model temperature (lower means more deterministic)
        
        Returns:
            Valid response string
        """
        try:
            if provider == "deepinfra":
                return await self.deepinfra.generate_with_constraint(
                    model=model,
                    messages=messages,
                    constraint=constraint,
                    max_attempts=max_attempts,
                    temperature=temperature
                )
            else:
                raise ValueError(f"Constrained generation not supported for provider: {provider}")
        except Exception as e:
            print(f"Error in LLMHelper.generate_with_constraint: {e}")
            raise
