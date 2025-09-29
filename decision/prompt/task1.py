import sys
import os

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from decision.prompt.format import prompt_template
from decision.utils.image_process import image_path_to_base64
from decision.utils.config_process import load_config


# Load YAML configuration file
config = load_config()
base_path = config["global"]["base_path"]
task_path = os.path.join(base_path, "asset", "decision")
# Get target object image path
target_item_image_path = os.path.join(task_path, "pic", "task1.png")

# Convert image to Base64 encoding
base64_image_data = image_path_to_base64(target_item_image_path)

# Define variables (excluding location and camera_name)
variables = {
    "target_item": "Mining Robot",
    "target_item_image": base64_image_data,
    "target_item_description": "A mining robot currently performing a task",
    "task_description": "Find the mining robot and report its exact location to the user",
    "another_target_photo": None
}

# Format the prompt
formatted_prompt = prompt_template.format(**variables)
print(formatted_prompt)

