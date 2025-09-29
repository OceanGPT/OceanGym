import sys
import os


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
target_item_image_path = os.path.join(task_path, "pic", "task6.png")

variables = {
    "target_item": "Wind power stations",
    "target_item_image": None,
    "target_item_description": "A power station built underwater with a surface component",
    "task_description": "Find the wind power stations and navigate to their base",
    "another_target_photo": None
}

formatted_prompt = prompt_template.format(**variables)
print(formatted_prompt)

