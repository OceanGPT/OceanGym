import os
import yaml

def load_config():

    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    print(f"? loaded config keys: {list(config.keys())}")
    return config

def remove_newlines_from_result(result):
    """
    Remove all newline characters from LLM result
    
    Args:
        result (str): The result string from ask_llm
        
    Returns:
        str: Result string with all newlines removed
    """
    if not result:
        return result
    
    # Remove all types of newline characters
    # \n - Unix/Linux newline
    # \r\n - Windows newline  
    # \r - Old Mac newline
    cleaned_result = result.replace('\n', '').replace('\r\n', '').replace('\r', '')
    
    return cleaned_result

def remove_newlines_preserve_spaces(result):
    """
    Remove newlines but preserve meaningful spaces between words
    
    Args:
        result (str): The result string from ask_llm
        
    Returns:
        str: Result string with newlines replaced by spaces
    """
    if not result:
        return result
    
    # Replace newlines with single space to preserve word separation
    cleaned_result = result.replace('\n', ' ').replace('\r\n', ' ').replace('\r', ' ')
    
    # Remove multiple consecutive spaces
    import re
    cleaned_result = re.sub(r'\s+', ' ', cleaned_result).strip()
    
    return cleaned_result