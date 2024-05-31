import pandas as pd
import json
import re

def load_data(file) -> pd.DataFrame:
    """
    Load data from a JSON file and return a DataFrame

    Parameters:
    file: bytes - The JSON file to load

    Returns:
    pd.DataFrame - The data in the JSON file as a DataFrame
    or None if there are any errors parsing the JSON file
    """
    data = json.loads(file)
    data = data["messages"]
    extracted_data = []
    for item in data:
        try:
            if item["type"] != "message":
                continue
            if item["text"] == "":
                continue
            name = item["from"]
            timestamp = item["date_unixtime"]
            content = remove_json_parts(item["text"])
            extracted_data.append({"name": name, "timestamp": int(timestamp), "content": content})
        except Exception:
            pass
    df = pd.DataFrame(extracted_data)
    if df.empty:
        return None
    return df

def remove_json_parts(string_data):
    json_strings = re.findall(r'\{.*\}|\[.*\]', string_data)  # Find potential JSON parts

    for json_string in json_strings:
        try:
            json.loads(json_string)  # Check if it's valid JSON
            string_data = string_data.replace(json_string, '')  # Replace with empty string if valid
        except json.JSONDecodeError:
            pass  # Ignore if not valid JSON
    return string_data
