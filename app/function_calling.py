import json
from typing import List, Dict, Any
from .llm_helper import LLMHelper

def create_function_selection_prompt(user_input: str, functions: List[Dict]) -> List[Dict[str, str]]:
    functions_summary = "\n".join(
        f"- {f['name']}: {f['description']}" for f in functions
    )
    
    return [
        {"role": "system", "content": """You are a helpful assistant that selects the most appropriate functions to call.
Your task is to return the function names that best match the user's request, separated by commas if multiple functions are needed.
If none of the available functions are suitable for the request, respond with 'no_need_to_call_any_known_function'.
Do not include any other text or explanation in your response."""},
        {"role": "system", "content": f"Available functions:\n{functions_summary}"},
        # Single function example
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": "get_weather"},
        # Multiple functions example
        {"role": "user", "content": "What's the weather in Paris and the stock price for AAPL?"},
        {"role": "assistant", "content": "get_weather,get_stock_price"},
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

class MultiEnumConstraint:
    """
    Validates that the model output is a comma-separated list of allowed values or no_need_to_call_any_known_function.
    Returns (is_valid, is_complete) tuple.
    """
    def __init__(self, valid_values: List[str]):
        self.valid_values = valid_values
    
    def __call__(self, text: str) -> tuple[bool, bool]:
        text = text.strip()
        if not text:
            return False, False
            
        if text == "no_need_to_call_any_known_function":
            return True, True
            
        functions = [f.strip() for f in text.split(',')]
        is_valid = all(f in self.valid_values for f in functions)
        return is_valid, True

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

async def process_function_call(client: LLMHelper, user_input: str, functions: List[Dict], completion_params: Dict) -> List[Dict] | None:
    # 1. Select functions
    messages = create_function_selection_prompt(user_input, functions)
    function_response = await client.generate_with_constraint(
        model=completion_params["llm_name"],
        messages=messages,
        constraint=MultiEnumConstraint([f["name"] for f in functions]),
        provider=completion_params.get('provider', 'deepinfra'),
        temperature=completion_params.get('temperature', 0.1),
        max_attempts=3
    )

    if function_response.strip() == "no_need_to_call_any_known_function":
        return None

    function_names = [name.strip() for name in function_response.split(',')]
    
    # Validate function names
    valid_functions = [f["name"] for f in functions]
    for function_name in function_names:
        if function_name not in valid_functions:
            raise ValueError(f"Invalid function name '{function_name}'. Must be one of: {valid_functions}")

    # 2. Generate arguments for each function
    result = []
    for function_name in function_names:
        messages = create_arguments_prompt(user_input, function_name, functions)
        arguments_response = await client.generate_with_constraint(
            model=completion_params["llm_name"],
            messages=messages,
            constraint=JsonSchemaConstraint(next(f for f in functions if f["name"] == function_name)["parameters"]),
            temperature=completion_params.get('temperature', 0.1),
            max_attempts=3
        )

        result.append({
            "name": function_name,
            "arguments": json.loads(arguments_response)
        })

    return result
