import base64
import os
import random
import re
from typing import List, Tuple

import numpy as np
import torch


def set_seed(seed: int):
    """set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_image_to_base64(image_path: str) -> str:
    """turn image to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def normalize_model_output(model_output: str, options: list[str]) -> list[str]:
    """
    Normalize the model output to a list of selected options.
    """
    def normalize(s: str) -> str:
        return re.sub(r"[-_\s]+", "", s.lower())

    option_map = {normalize(opt): opt for opt in options}

    tokens = re.findall(r"[a-zA-Z0-9_-]+", model_output)

    selected = []
    for token in tokens:
        key = normalize(token)
        if key in option_map and option_map[key] not in selected:
            selected.append(option_map[key])

    return selected


def compare_eval(
        options: List[str],
        model_output: List[str],
        goal: List[str],
    ) -> Tuple[bool, int]:
    """
    Compare the model output with the target option and return (whether they match exactly, the number of matches)
    """
    # Convert to Lowercase Set
    model_set = set([x.lower() for x in model_output])
    goal_set = set([x.lower() for x in goal])

    # Is it a perfect match
    is_completely_match = model_set == goal_set

    # matching quantity
    matched_count = len(model_set & goal_set)

    return is_completely_match, matched_count
