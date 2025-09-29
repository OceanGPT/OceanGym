import sys
import os

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from decision.prompt.format import prompt_template
from decision.utils.image_process import image_path_to_base64
from decision.utils.config_process import load_config

# Load YAML configuration file
config = load_config()
base_path = config["global"]["base_path"]
task_path = os.path.join(base_path, "asset", "decision")
# Get target object image path
target_item_image_path = os.path.join(task_path, "pic", "task2.png")


# Define variables (excluding location and camera_name)
variables = {
    "target_item": "Oil pipelines",
    "target_item_image": image_path_to_base64(target_item_image_path),
    "target_item_description": "A long oil pipeline with multiple valves, located on the seabed, partially covered by sand",
    "task_description": "Find the pipeline, report its exact location, explore along the pipeline and record the number of valves",
    "another_target_photo": None
}

# Format the prompt
formatted_prompt = prompt_template.format(**variables)