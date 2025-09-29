import yaml
import logging
from datetime import datetime
from openai import OpenAI
import re  # For regular expression parsing
import numpy as np
import json
import os



# Set maximum length for memory
MEMORY_MAX_LENGTH = 1000 # Can be adjusted as needed
IMPORTANT_MEMORY_MAX_LENGTH = 500  # Maximum length for important memory
memory = []  # Regular memory list
important_memory = []  # Important memory list

def save_memory_to_file(memory_data, filename):
    """
    Save memory data to JSON file
    :param memory_data: Memory data to save
    :param filename: Target file name
    """
    try:
        # Convert numpy arrays to lists before saving
        serializable_data = convert_ndarray_to_list(memory_data.copy())
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Memory saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving memory to {filename}: {e}")

def load_memory_from_file(filename):
    """
    Load memory data from JSON file
    :param filename: Source file name
    :return: Loaded memory data or empty list if file doesn't exist
    """
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Memory loaded from {filename}")
            return data
        else:
            logging.info(f"Memory file {filename} not found, starting with empty memory")
            return []
    except Exception as e:
        logging.error(f"Error loading memory from {filename}: {e}")
        return []


def save_all_memory(mem_file, important_mem_file, memory_data, important_memory_data):
    """
    Save both regular and important memory to separate files
    :param mem_dir: Directory to save memory files
    :param memory_data: Regular memory data
    :param important_memory_data: Important memory data
    """
    try:
        
        save_memory_to_file(memory_data, mem_file)
        save_memory_to_file(important_memory_data, important_mem_file)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"All memory saved successfully to {mem_file} at {current_time}")
        print(f"All memory saved successfully to {mem_file} at {current_time}")
        
    except Exception as e:
        logging.error(f"Error in save_all_memory: {e}")
        print(f"Error in save_all_memory: {e}")

def update_memory(current_memory,target_info, action, location):
    """
    Update Memory with target information, action, and location
    :param target_info: Target object information (e.g., LLM output)
    :param action: Current action (e.g., command)
    :param location: Current location (e.g., camera name or coordinates)
    :param base64_image: Image of the target in base64 format
    """
    # Handle action conversion
        
    # Handle location conversion
    if isinstance(location, np.ndarray):
        location_list = location.tolist()
    else:
        location_list = location
    
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_info": target_info,
        "action": action,
        "location": location_list
    }
    current_memory.append(record)

    # If memory exceeds maximum length, remove the oldest record
    if len(current_memory) > MEMORY_MAX_LENGTH:
        current_memory.pop(0)  # Remove the oldest record
    print("length of mem is ", len(current_memory))
    logging.info(f"Memory updated: {record}")


def update_important_memory(current_important_memory,target_info, action, location):
    """
    Update Important Memory with target information, action, and location
    :param target_info: Target object information (e.g., LLM output)
    :param action: Current action (e.g., command)
    :param location: Current location (e.g., camera name or coordinates)
    :param base64_image: Image of the target in base64 format
    """
    # Handle action conversion

        
    # Handle location conversion
    if isinstance(location, np.ndarray):
        location_list = location.tolist()
    else:
        location_list = location
    
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_info": target_info,
        "action": action,
        "location": location_list
    }
    current_important_memory.append(record)

    # If important_memory exceeds maximum length, remove the oldest record
    if len(current_important_memory) > IMPORTANT_MEMORY_MAX_LENGTH:
        current_important_memory.pop(0)  # Remove the oldest record

    logging.info(f"Important Memory updated: {record}")
    


# Convert numpy.ndarray to list in memory
def convert_ndarray_to_list(memory_data):
    """
    Convert numpy arrays to lists in memory data
    :param memory_data: Memory data to convert
    :return: Converted memory data
    """
    if not isinstance(memory_data, list):
        return memory_data
        
    converted_data = []
    for record in memory_data:
        converted_record = {}
        for key, value in record.items():
            if isinstance(value, np.ndarray):  # Check if it's numpy.ndarray
                converted_record[key] = value.tolist()  # Convert to Python list
            else:
                converted_record[key] = value
        converted_data.append(converted_record)
    return converted_data

def extract_mem_from_output(output):
    """
    Extract content wrapped in ##...## from the LLM output
    :param output: Complete LLM output
    :return: Extracted content list
    """
    # extract and clean matches
    matches = re.findall(r"\#\#(.*?)\#\#", output, re.DOTALL)
    # clean and filter matches
    cleaned_matches = []
    for match in matches:
        cleaned = match.strip()  
        cleaned = ' '.join(cleaned.split())  
        if cleaned:  
            cleaned_matches.append(cleaned)
    return cleaned_matches

def extract_important_mem_from_output(output):
    """
    Extract important content wrapped in $$....$$ from the LLM output
    :param output: Complete LLM output
    :return: Extracted content list
    """
    matches = re.findall(r"\$\$(.*?)\$\$", output, re.DOTALL)
    cleaned_matches = []
    for match in matches:
        cleaned = match.strip()
        cleaned = ' '.join(cleaned.split())
        if cleaned:
            cleaned_matches.append(cleaned)
    return cleaned_matches

def clear_memory():
    """
    Clear all memory data
    """
    global memory, important_memory
    memory = []
    important_memory = []
    logging.info("Memory cleared")


