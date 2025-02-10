import os
import aiohttp
import logging
from typing import List, Dict

class DeepInfraHelper:
    def __init__(self):
        self.api_token = os.getenv('DEEPINFRA_API_KEY')
        if not self.api_token:
            raise ValueError("DEEPINFRA_API_KEY environment variable not set")
        
        self.base_url = "https://api.deepinfra.com/v1/openai/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }

    async def get_completion(self, messages, model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", max_new_tokens=500, temperature=0.5, top_p=1, stream=False):
        """
        Get completion from DeepInfra API
        
        Args:
            messages (list): List of message dictionaries with role and content
            model (str): The model to use for completion
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0-1)
            top_p (float): Controls diversity via nucleus sampling (0-1)
            stream (bool): Whether to stream the response
            
        Returns:
            str: The model's response
        """
        try:
            logging.info(f"Using model: {model}")

            data = {
                "messages": messages,
                "model": model,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }

            logging.debug(f"Sending request to DeepInfra API with data: {data}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=self.headers,
                    json=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"DeepInfra API error: {response.status}")
                        logging.error(f"Response content: {error_text}")
                        response.raise_for_status()
                    
                    response_json = await response.json()
                    return response_json
            
        except aiohttp.ClientError as e:
            logging.error(f"Request error in get_completion: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in get_completion: {str(e)}")
            raise

    async def generate_with_constraint(
        self,
        model: str,
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
            response = await self.get_completion(
                model=model,
                messages=messages,
                temperature=temperature
            )
            result = response['choices'][0]['message']['content']
            valid, complete = constraint(result)
            
            if valid and complete:
                return result
            
            # Increase temperature slightly on retries to get different responses
            temperature += 0.1
            
        raise ValueError(f"Failed to generate valid response after {max_attempts} attempts")

if __name__ == "__main__":
    # Simple test functionality
    import asyncio
    
    async def main():
        helper = DeepInfraHelper()
        
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        
        try:
            response = await helper.get_completion(
                messages=messages,
                max_new_tokens=2000,
                temperature=0.7,
                top_p=0.9
            )
            print("Response:")
            print("-" * 80)
            print(response)
            print("-" * 80)
            
        except Exception as e:
            print(f"Error occurred: {e}")

    asyncio.run(main())
