# [This API is available at DuckHosting.lol](https://www.duckhosting.lol/)
# [Discord for Support](https://discord.com/invite/NYpwUh7KNS)

## Usage Examples

### Example 1: Weather Query
**User:** I never was in hawaii during summer, I wonder how it feels?

**Response:**
```json
{
    "function": "get_weather",
    "arguments": {
        "location": "Hawaii",
        "season": "summer"
    }
}
```

### Example 2: Stock Query
**User:** I never bought rivian stocks from revolut, may ask some more info about them?

**Response:**
```json
{
    "function": "get_stock_price",
    "arguments": {
        "stock_name": "RIVN",
        "broker_name": "Revolut"
    }
}
```

### Example 3: Multiple Functions
**User:** I was once in hawaii during summer and was buying rivian stocks there using revolut, I wonder how is it all now?

**Response:**
```json
[
    {
        "function": "get_weather",
        "arguments": {
            "location": "Hawaii",
            "season": "summer"
        }
    },
    {
        "function": "get_stock_price",
        "arguments": {
            "stock_name": "Rivian",
            "broker_name": "Revolut"
        }
    }
]
```

### Example 4: Invalid Query
**User:** I would like to eat an apple pie

**Response:**
```
Error in LLMHelper.generate_with_constraint: Failed to generate valid response after 3 attempts
Error: Failed to generate valid response after 3 attempts
```

Told llm to make a short description of parent repo and 
then recreated it from description. Just couldn't understand anything from that documentation.

# Function Calling Implementation Guide

This guide explains the core components and flows for implementing function calling with LLMs.

## 1. Function Description Format

Functions are described using a JSON schema format:

```python
function_description = {
    "name": "function_name",
    "description": "What the function does",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Parameter description"
            },
            # ... other parameters
        }
    }
}
```

## 2. Core Flow Components

### 2.1 Function Selection Flow

1. Create a prompt combining:
   - System instruction to select a function
   - List of available functions with descriptions
   - User's input
   - Response prefix

```python
prompt = f"""[INST] <<SYS>>
Help choose the appropriate function to call to answer the user's question.

Available functions:
{functions_summary}
<</SYS>>

{user_input} [/INST] Here's the function the user should call: """
```

2. Send to model and enforce response to be one of the function names

### 2.2 Function Arguments Generation Flow

1. Create a prompt combining:
   - System instruction to generate arguments
   - Function description and parameters schema
   - User's input
   - JSON response prefix

```python
prompt = f"""[INST] <<SYS>>
Define the arguments for {function_name} to answer the user's question.

Function description: {function_description}
Function parameters should follow this schema:
```jsonschema
{parameters_schema}
```
<</SYS>>

{user_input} [/INST] Here are the arguments for the `{function_name}` function: ```json
"""
```

2. Send to model and enforce JSON schema validation on the response

## 3. Key Implementation Components

### 3.1 JSON Schema Constraint

Use JSON schema validation to ensure model outputs valid function arguments:

```python
class JsonSchemaConstraint:
    def __init__(self, schema: dict):
        self.parser = json_schema_enforcer.parser_for_schema(schema)
    
    def validate(self, text: str) -> tuple[bool, bool]:
        result = self.parser.validate(text)
        return result.valid, result.end_index is not None
```

### 3.2 Generation Control

Control token generation to ensure valid responses:

```python
def generate_with_constraint(
    model,
    prompt: str, 
    constraint: Callable[[str], tuple[bool, bool]]
) -> str:
    generation = model.start_generation(prompt)
    
    while True:
        for token in generation.get_sorted_tokens():
            candidate = generation.get_generated(token)
            is_valid, is_complete = constraint(candidate)
            
            if is_valid:
                generation.register_token(token)
                if is_complete:
                    return candidate
                break
```

## 4. Usage Example

```python
# Define functions
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
        }
    }
}]

# 1. Function Selection
function_name = generate_with_constraint(
    model,
    create_function_selection_prompt(user_input, functions),
    EnumConstraint([f["name"] for f in functions])
)

# 2. Generate Arguments
if function_name:
    arguments_json = generate_with_constraint(
        model,
        create_arguments_prompt(user_input, function_name, functions),
        JsonSchemaConstraint(get_function_schema(function_name, functions))
    )
    
    # 3. Execute function
    function_call = {
        "name": function_name,
        "arguments": arguments_json
    }
    # Call your function implementation here
```

## 5. Implementation Notes

- Use token-by-token generation to maintain control over the output format
- Implement proper error handling for invalid responses
- Consider adding retry logic for failed generations
- Cache function descriptions and schemas for better performance
- Consider adding temperature and other generation parameters control
