import json
import requests
from typing import List, Dict, Any

class DeepInfraClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepinfra.com/v1/openai/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": messages,
            "temperature": temperature
        }
        
        print("\n=== LLM Request ===")
        print("Messages:")
        for msg in messages:
            print(f"{msg['role'].upper()}: {msg['content'][:200]}...")
        print(f"Temperature: {temperature}")
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()["choices"][0]["message"]["content"]
        print("\n=== LLM Response ===")
        print(result)
        print("==================\n")
        
        return result

    def generate_with_constraint(
        self,
        messages: List[Dict[str, str]],
        constraint: callable,
        max_attempts: int = 3,
        temperature: float = 0.1
    ) -> str:
        """
        Generate a response that satisfies the given constraint.
        
        Args:
            messages: List of conversation messages
            constraint: Callable that returns (is_valid, is_complete) tuple
            max_attempts: Maximum number of retries for invalid responses
            temperature: Model temperature (lower means more deterministic)
        
        Returns:
            Valid response string
            
        Raises:
            ValueError: If no valid response after max_attempts
        """
        for attempt in range(max_attempts):
            result = self.generate(messages, temperature)
            valid, complete = constraint(result)
            
            if valid and complete:
                return result
            
            # Increase temperature slightly on retries to get different responses
            temperature += 0.1
            
        raise ValueError(f"Failed to generate valid response after {max_attempts} attempts")
