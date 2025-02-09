import json
from typing import List, Dict, Any
from deep_infra_client import DeepInfraClient

def create_function_selection_prompt(user_input: str, functions: List[Dict]) -> List[Dict[str, str]]:
    functions_summary = "\n".join(
        f"- {f['name']}: {f['description']}" for f in functions
    )
    
    # Adding a comment to explain the few-shot learning format
    return [
        {"role": "system", "content": """You are a helpful assistant that selects the most appropriate function to call.
Your task is to return ONLY the function name that best matches the user's request.
Do not include any other text or explanation in your response."""},
        {"role": "system", "content": f"Available functions:\n{functions_summary}"},
        # Example interaction to demonstrate desired behavior
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": "get_weather"},  # Example of correct response format
        # Actual user query
        {"role": "user", "content": user_input}
    ]

def create_arguments_prompt(user_input: str, function_name: str, functions: List[Dict]) -> List[Dict[str, str]]:
    function = next(f for f in functions if f["name"] == function_name)
    
    return [
        {"role": "system", "content": f"""You are a helpful assistant that generates function arguments in JSON format.
Your task is to extract relevant information from the user's request and format it as JSON.
Respond ONLY with a valid JSON object matching this schema:
{json.dumps(function['parameters'], indent=2)}

Example response for '{function_name}':
{{"location": "Paris"}}"""},
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "content": '{"location": "Tokyo"}'},
        {"role": "user", "content": user_input}
    ]

class EnumConstraint:
    """
    Validates that the model output is one of the allowed values.
    Returns (is_valid, is_complete) tuple.
    """
    def __init__(self, valid_values: List[str]):
        self.valid_values = valid_values
    
    def __call__(self, text: str) -> tuple[bool, bool]:
        text = text.strip()
        # Return (is_valid, is_complete)
        # is_valid: text matches one of the valid values
        # is_complete: always True since we only need exact matches
        return text in self.valid_values, True

class JsonSchemaConstraint:
    """
    Validates that the model output is valid JSON matching the schema.
    Returns (is_valid, is_complete) tuple.
    """
    def __init__(self, schema: Dict):
        self.schema = schema
    
    def __call__(self, text: str) -> tuple[bool, bool]:
        try:
            data = json.loads(text)
            # Basic validation of required fields
            if "required" in self.schema:
                for field in self.schema["required"]:
                    if field not in data:
                        return False, True
            # Return (is_valid, is_complete)
            # is_valid: JSON is valid and matches schema
            # is_complete: always True since we have full JSON
            return True, True
        except json.JSONDecodeError:
            # JSON is invalid, but might be incomplete
            return False, False

def process_function_call(client: DeepInfraClient, user_input: str, functions: List[Dict]) -> Dict:
    # 1. Select function
    messages = create_function_selection_prompt(user_input, functions)
    function_name = client.generate_with_constraint(
        messages,
        EnumConstraint([f["name"] for f in functions])
    )

    # 2. Generate arguments
    messages = create_arguments_prompt(user_input, function_name, functions)

    print("=================")
    print("messages", messages)
    print("=================")
    arguments = client.generate_with_constraint(
        messages,
        JsonSchemaConstraint(next(f for f in functions if f["name"] == function_name)["parameters"])
    )

    return {
        "name": function_name,
        "arguments": json.loads(arguments)
    }
