import asyncio
from .function_calling import process_function_call
from .llm_helper import LLMHelper
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def main():
    functions = [{
        "name": "get_weather",
        "description": "Get weather in a location for a specific season",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "season": {
                    "type": "string",
                    "description": "Season of the year (spring, summer, fall, winter, or current)",
                    "enum": ["spring", "summer", "fall", "winter", "current"]
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_stock_price",
        "description": "Get stock price from a specific broker",
        "parameters": {
            "type": "object",
            "properties": {
                "stock_name": {
                    "type": "string",
                    "description": "Stock symbol or name"
                },
                "broker_name": {
                    "type": "string",
                    "description": "Name of the broker"
                }
            },
            "required": ["stock_name", "broker_name"]
        }
    }]

    llm = LLMHelper()
    
    while True:
        user_input = input("Ask about weather or stocks (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        try:
            completion_params = {
                "provider": "deepinfra",
                "llm_name": "deepseek-ai/DeepSeek-V3",
                "temperature": 0.7,
                "max_new_tokens": 2048,
                "top_p": 0.9
            }
            
            function_calls = await process_function_call(llm, user_input, functions, completion_params)
            if function_calls is None:
                print("\nNo known function calls needed for this request.")
            else:
                for call in function_calls:
                    print(f"\nFunction: {call['name']}")
                    print(f"Arguments: {call['arguments']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
