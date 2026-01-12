import json
import os
from dotenv import load_dotenv
from pathlib import Path
import argparse
load_dotenv(".env")

# Map of providers base URLs
PROVIDER_BASE_URLS = {
    "sambanova": "https://api.sambanova.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1"
}

# Map of providers api keys
PROVIDER_API_KEYS = {
    "sambanova": os.getenv("SAMBANOVA_API_KEY", "**************"),
    "groq": os.getenv("GROQ_API_KEY", "**************"),
    "cerebras": os.getenv("CEREBRAS_API_KEY", "**************"),
    "together": os.getenv("TOGETHER_API_KEY", "**************"),
    "fireworks": os.getenv("FIREWORKS_API_KEY", "**************")
}

def extract_provider_from_model_name(model_name):
    """
    Extracts the provider name from a model_name string like 'sambanova_Meta-Llama-3.3-70B-Instruct-FC'.
    Assumes provider is the prefix before the first underscore.
    """
    if "_" in model_name:
        return model_name.split("_")[0].lower()
    return None

def convert_float_to_number(schema):
    """
    Recursively converts all 'float' types to 'number' in a JSON schema.
    """
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "type" and value == "float":
                schema[key] = "number"
            else:
                convert_float_to_number(value)
    elif isinstance(schema, list):
        for item in schema:
            convert_float_to_number(item)
            
def convert_tuple_to_array(schema):
    """
    Recursively converts all 'tuples' types to 'array' in a JSON schema.
    """
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "type" and value == "tuple":
                schema[key] = "array"
            else:
                convert_tuple_to_array(value)
    elif isinstance(schema, list):
        for item in schema:
            convert_tuple_to_array(item)
            
def convert_dict_to_object(schema):
    """
    Recursively converts all 'dict' types to 'object' in a JSON schema.
    """
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "type" and value == "dict":
                schema[key] = "object"
            else:
                convert_dict_to_object(value)
    elif isinstance(schema, list):
        for item in schema:
            convert_dict_to_object(item)
            
def normalize_model_name(model_name):
    # Remove provider prefix (before first underscore)
    if "_" in model_name:
        model_name = model_name.split("_", 1)[1]
    # Remove "-FC" suffix if present
    if model_name.endswith("-FC"):
        model_name = model_name[:-3]
    # Replace underscores with slashes
    return model_name.replace("_", "/")

def escape_json_for_curl(json_string: str) -> str:
    """
    Escapes a JSON string to be safely embedded in a curl --data single-quoted string.
    Replaces single quotes with '\'' which closes the quote, adds an escaped quote, then reopens it.
    """
    return json_string.replace("'", "'\"'\"'")

def generate_curl_command(data, provider=None, model_name=None):
 # Always get the raw model name from the JSON (to extract provider)
    raw_model_name = data.get("model_name")
    if not raw_model_name and not model_name:
        raise ValueError("Model name must be provided or found in the JSON.")
    
    # Determine the model name to use in the request
    final_model_name = normalize_model_name(model_name or raw_model_name)
    
    # Determine provider: from argument, or from raw_model_name
    if provider is None:
        provider = extract_provider_from_model_name(raw_model_name or "")

    if provider not in PROVIDER_BASE_URLS:
        raise ValueError(f"Unknown or missing provider: '{provider}'")
    
    # Determine base URL
    base_url = PROVIDER_BASE_URLS[provider]
    bearer_token = PROVIDER_API_KEYS[provider]
    
    # Extract user message
    message_content = data['prompt']['question'][0][0]['content']
    
    # Extract tools from prompt
    tools = []
    for func in data['prompt'].get('function', []):
        convert_float_to_number(func)  # Fix float → number
        convert_tuple_to_array(func) # Fix tuple → array
        convert_dict_to_object(func) # Fix dict → object
        tools.append({
            "type": "function",
            "function": func
        })

    # Construct the payload
    payload = {
        "model": final_model_name,
        "messages": [
            {
                "role": "user",
                "content": message_content
            }
        ],
        "tools": tools,
        "temperature": 0.0,
        "response_format": None,
        "stream": False
    }
    json_data = json.dumps(payload, indent=4)
    escaped_data = escape_json_for_curl(json_data)

    # Build the curl command
    curl_command = f"""curl --location '{base_url}/chat/completions' \\
--header 'Content-Type: application/json' \\
--header 'Authorization: Bearer {bearer_token}' \\
--data '{escaped_data}'"""

    return curl_command

def load_sample_json(run_id, raw_model_name, sample_id):
    """
    Load a specific sample from a JSONL file given provider/run_id/model/test_type/sample_id.
    """
    provider = extract_provider_from_model_name(raw_model_name)
    test_type = "_".join(sample_id.split("_")[:-1])
    file_path = (
        Path("score") /
        provider /
        run_id /
        raw_model_name /
        f"BFCL_v3_{test_type}_score.json"
    )

    if not file_path.exists():
        raise FileNotFoundError(f"Score file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("id") == sample_id:
                return entry

    raise ValueError(f"Sample ID '{sample_id}' not found in {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate a curl command from base gorilla tests")
    parser.add_argument('--run_id', help="optional run id (date folder name)")
    parser.add_argument('--raw_model_name', help="raw model name for gorilla run <provider>_<model_name>-FC")
    parser.add_argument('--sample_id', help="test sample id <test_category>_<index>")
    parser.add_argument('--json',  help="optional json input")
    parser.add_argument('--provider', help="Optional provider override (sambanova, groq, etc).")
    parser.add_argument('--model_name', help="Optional model name override.")
    args = parser.parse_args()

    if args.json:
        curl_cmd = generate_curl_command(data=json, provider=args.provider, model_name=args.model_name)
        print("\nGenerated cURL command:\n")
        print(curl_cmd)
    elif args.run_id and args.raw_model_name and args.sample_id:
        sample_json = load_sample_json(
            run_id=args.run_id,
            raw_model_name=args.raw_model_name,
            sample_id=args.sample_id
        )
        curl_cmd = generate_curl_command(data=sample_json)
        print("\nGenerated cURL command:\n")
        print(curl_cmd)
    else:
        sample_json = load_sample_json(
            run_id="2025-06-19-23-36-53",
            raw_model_name="sambanova_Meta-Llama-3.3-70B-Instruct-FeC",
            sample_id="parallel_multiple_151"
            )
        #sample_json = {"id": "parallel_multiple_31", "model_name": "sambanova_Meta-Llama-3.3-70B-Instruct-FC", "test_category": "parallel_multiple", "valid": False, "error": ["Invalid syntax. Failed to decode AST. 'str' object has no attribute 'keys'"], "error_type": "ast_decoder:decoder_failed", "prompt": {"id": "parallel_multiple_31", "question": [[{"role": "user", "content": "Find how many cases and the judge handling a specific lawsuit for Pacific Gas and Electric and Tesla Inc."}]], "function": [{"name": "lawsuit.fetch_details", "description": "Fetch the details of a lawsuit for a specific company.", "parameters": {"type": "dict", "properties": {"company_name": {"type": "string", "description": "The company involved in the lawsuit."}}, "required": ["company_name"]}}, {"name": "lawsuit.judge", "description": "Fetch the judge handling a lawsuit for a specific company.", "parameters": {"type": "dict", "properties": {"company_name": {"type": "string", "description": "The company involved in the lawsuit."}, "lawsuit_id": {"type": "float", "description": "The ID number of the lawsuit. Default to 123", "default": 123}}, "required": ["company_name"]}}]}, "model_result_raw": "Error during inference: 'NoneType' object is not subscriptable", "possible_answer": [{"lawsuit.fetch_details": {"company_name": ["Pacific Gas and Electric", "PG&E"]}}, {"lawsuit.judge": {"company_name": ["Pacific Gas and Electric", "PG&E"], "lawsuit_id": [123, ""]}}, {"lawsuit.fetch_details": {"company_name": ["Tesla Inc.", "Tesla"]}}, {"lawsuit.judge": {"company_name": ["Tesla Inc.", "Tesla"], "lawsuit_id": [123, ""]}}]}
        curl = generate_curl_command(data=sample_json)#, provider="sambanova", model_name="Meta-Llama-3.3-70B-Instruct")

        print(curl)
if __name__ == "__main__":
    main()
    
    # sample usage:
    # python make_gorilla_curl.py --run_id 2025-06-19-23-36-53 --raw_model_name sambanova_Meta-Llama-3.3-70B-Instruct-FC --sample_id parallel_multiple_151