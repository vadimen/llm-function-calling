from function_calling import process_function_call
from deep_infra_client import DeepInfraClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("DEEP_INFRA")

if not API_KEY:
    raise ValueError("DEEP_INFRA API key not found in .env file")

def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny"

# Example prompt to show the user what kind of questions they can ask
EXAMPLE_PROMPTS = [
    "What's the weather in Paris?",
    "Tell me the weather in Tokyo",
    "How's the weather in New York?",
    "Check weather London"
]

def main():
    functions = [{
        "name": "get_weather",
        "description": "Get weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }]

    client = DeepInfraClient(API_KEY)
    
    print("\nExample questions you can ask:")
    for example in EXAMPLE_PROMPTS:
        print(f"- {example}")
    print()
    
    while True:
        user_input = input("Ask about weather (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        try:
            function_call = process_function_call(client, user_input, functions)
            if function_call["name"] == "get_weather":
                result = get_weather(**function_call["arguments"])
                print(result)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
