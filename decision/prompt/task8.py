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
target_item_image_path = os.path.join(task_path, "pic", "task8.png")

# Convert image to Base64 encoding
base64_image_data = image_path_to_base64(target_item_image_path)

# Define variables (excluding location and camera_name)
variables = {
    "target_item": "Landing platforms with \"H\" markings",
    "target_item_image": base64_image_data,
    "target_item_description": "A landing platform used for robot charging, likely located near wind turbines",
    "task_description": "Find the landing platform with \"H\" markings and LAND SAFELY on it. The primary goal is precise landing - continuously adjust your position to keep the platform EXACTLY in the CENTER of your camera view while descending. Make small, careful movements to maintain this center alignment throughout the approach. After successful landing, report the exact location to the user. IMPORTANT: The mission is only considered complete after successful landing with the platform kept centered in your camera view, not just finding the platform. After landing, the robot should stop all movement to complete the docking procedure.",
    "another_target_photo": None
}

# Format the prompt
formatted_prompt = prompt_template.format(**variables)


# Target coordinates: -71 151 -53