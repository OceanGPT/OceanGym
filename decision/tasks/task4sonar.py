import json
import sys
import os

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import threading
import logging
import matplotlib.pyplot as plt
import holoocean
import numpy as np
from pynput import keyboard
import cv2
from datetime import datetime, timedelta

from decision.utils.command import (
    start_keyboard_listener, parse_keys, pressed_keys, parse_llm_output, parse_action_from_llm
)
from decision.llm.llm import ask_llm, found_target, get_llm_info
from decision.llm.memory import (
    extract_mem_from_output, update_memory, memory, important_memory,
    convert_ndarray_to_list, extract_important_mem_from_output,
    update_important_memory, save_all_memory, clear_memory, load_memory_from_file
)
from decision.utils.config_process import load_config, remove_newlines_preserve_spaces
from decision.utils.image_process import image_to_base64_128, image_to_base64_256, process_camera_output
from decision.prompt.task4 import formatted_prompt

# Load YAML configuration file
config = load_config()

# Get base and task paths from config
base_path = config["global"]["base_path"]
task_path = os.path.join(base_path, "task", "decision")

# Define prompt and log/memory file paths
prompt_file = os.path.join(task_path, "prompt", "task4.py")
log_dir = os.path.join(base_path, "outputs", "decision", "log", "log_4")
mem_dir = os.path.join(base_path, "outputs", "decision", "memory", "mem_4")
time_limit = config["defaults"]["time"]

# Get other parameters from config
camera_keys = config["defaults"]["camera_keys"]
scenario_name = config["scenario"]["name"]
llm_interval = config["defaults"]["interval"]
llm_mode, llm_model, llm_name = get_llm_info(config)

# Initialize log directory
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # Create log directory
if not os.path.exists(mem_dir):
    os.makedirs(mem_dir)  # Create memory directory

# Use current timestamp to name log file
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"llm_output_{llm_model}_{current_time}.log")
mem_file = os.path.join(mem_dir, f"memory_{current_time}.json")
important_mem_file = os.path.join(mem_dir, f"important_memory_{current_time}.json")

# Initialize logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
print(f"Log file path: {os.path.abspath(log_file)}")

# Global variables
cached_command = np.zeros(8)  # Cached LLM actions
llm_lock = threading.Lock()  # For thread safety
llm_running = False  # Whether an LLM request is running

def load_prompt(file_path):
    """
    Load prompt content from file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

prompt = load_prompt(prompt_file)
counter = 0
pic_counter = 1
config = holoocean.packagemanager.get_scenario(scenario_name)
find_target = False  # Whether the target object is found

with holoocean.make(scenario_cfg=config, start_world=False) as env:
    llm_mode = False
    command = np.zeros(8)  # Initial no action
    cached_command = np.zeros(8)  # Cached LLM action

print("Press 'm' to enter LLM control mode, press 'q' to exit.")
start_keyboard_listener()
clear_memory()
print(formatted_prompt)
start_time = datetime.now()

while True:
    current_time = datetime.now()
    elapsed_time = current_time - start_time

    # Check if runtime exceeds time limit
    if elapsed_time > timedelta(minutes=time_limit):
        print(f"Runtime exceeded {time_limit} minutes, program exiting.")
        logging.error(f"Runtime exceeded {time_limit} minutes, program exiting.")
        break
    # Check for quit command
    if 'q' in pressed_keys:
        break

    # Toggle LLM mode
    if 'm' in pressed_keys:
        llm_mode = not llm_mode
        pressed_keys.remove('m')
        print(f'LLM mode: {llm_mode}')

    if llm_mode:
        counter += 1
        state = env.tick()
        # Use cached LLM action for non-inference frames
        env.act('auv0', cached_command)

        use_llm = False
        b64_images = []  # Store Base64 images of all cameras

        # Process camera output for display
        process_camera_output(state)

        # Check if all cameras have data
        camera_keys = ["LeftCamera", "RightCamera", "DownCamera", "UpCamera", "FrontCamera", "BackCamera"]

        base64_images = []
        for camera in camera_keys:
            if camera in state:
                pixels = state[camera]
                if pixels is None or pixels.size == 0:
                    print(f"{camera} data is empty, skipping")
                    continue

                # Display camera output
                cv2.imshow(f"{camera} Output", pixels[:, :, 0:3])
                cv2.waitKey(1)

                # Convert to Base64 and add to list
                if(camera=="DownCamera"):
                    image_base64 = image_to_base64_256(pixels)
                else:
                    image_base64 = image_to_base64_128(pixels)
                b64_images.append(image_base64)
        sonar_b64_image = image_to_base64_256(state["ImagingSonar"])
        # If all cameras have data, prepare for LLM call
        if b64_images:
            pic_counter += 1
            use_llm = True

        # Call LLM every 4 frames to reduce computational load
        if use_llm and pic_counter % 4 == 0:

            # Step 1: Reset and build basic information
            prompt = formatted_prompt
            prompt += f"If you detect that the surrounding visibility is low, please prioritize using sonar data for analysis.Sonar Image:\n$$\n{sonar_b64_image}\n$$\n"

            loc = state["Location"]
            loc_formatted = [round(float(coord), 1) for coord in loc]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prompt += f"Current time: {current_time}\n"
            prompt += f"Current position: {loc_formatted}\n"

            # Step 2: Prepare and concatenate historical memory (must be done before calling ask_llm)
            current_mem = convert_ndarray_to_list(memory)
            current_important_mem = convert_ndarray_to_list(important_memory)

            # Add memory guidance header
            prompt += """
---
**[Historical Memory Logs for Tactical Review]**
To inform your decision, review your operational logs below. `memory` contains your recent routine observations, while `important_memory` contains mission-critical findings.

**Standard Memory Log:**
"""
            prompt += json.dumps(current_mem, indent=2, ensure_ascii=False)

            prompt += """

**Critical Memory Log:**
"""
            prompt += json.dumps(current_important_mem, indent=2, ensure_ascii=False)
            prompt += "\n---\n"

            # Step 3: Append final task command (this is the last text part to be added)
            prompt += """
**[Your Task]**
As the Control Expert, based on your analysis of the provided real-time images and historical memory logs, execute your decision cycle according to the following strict protocol:
1.  **PRIMARY ANALYSIS - TARGET ACQUISITION:** Scrutinize the six (6) non-target, real-time camera feeds provided. Is the target object visible in any of them?
2.  **DECISION & ACTION:**
    -   **IF TARGET IS VISIBLE:** You must immediately initiate the 'Target ' protocol. Report using the "$$...$$" format and issue commands to approach and position over the target.
    -   **IF TARGET IS NOT VISIBLE:** You must initiate the 'Exploration' protocol. Analyze the current environment from the feeds, report it using the "##...##" format, then consult your text-based memory logs to inform your strategy, and finally issue the next optimal command to continue the search.
"""

            # Step 4: Everything is ready, call the LLM
            result = ask_llm(prompt, b64_images)

            # Step 5: Process the returned result
            command = parse_llm_output(result)
            cached_command = command
            logging.info(f"LLM output: {result}")

            # Check if target is found and update memory
            find_target = found_target(result)
            target_info = extract_mem_from_output(result)
            action = parse_action_from_llm(result)
            update_memory(memory, target_info, action, loc_formatted)

            # If target is found, update important memory
            if find_target:
                print("Target object found")
                logging.info(f"Target object found at location {loc_formatted}")
                target_info = extract_important_mem_from_output(result)
                action = parse_action_from_llm(result)
                update_important_memory(important_memory, target_info, action, loc_formatted)

            # Save all memory to files
            save_all_memory(mem_file, important_mem_file, memory, important_memory)
            print("Important information: " + str(important_memory))
            use_llm = False
    else:
        # Non-LLM mode: keyboard manual control
        command = parse_keys(pressed_keys)
        env.act('auv0', command)
        state = env.tick()
        process_camera_output(state)

plt.ioff()
plt.show()