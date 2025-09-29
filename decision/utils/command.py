import numpy as np
from pynput import keyboard

# For keyboard control
pressed_keys = []
force1 = 100
force2 = 5

def on_press(key):
    """
    Keyboard press event
    """
    global pressed_keys
    try:
        if hasattr(key, 'char') and key.char not in pressed_keys:
            pressed_keys.append(key.char)
    except AttributeError:
        pass  # Ignore special keys

def on_release(key):
    """
    Keyboard release event
    """
    global pressed_keys
    try:
        if hasattr(key, 'char') and key.char in pressed_keys:
            pressed_keys.remove(key.char)
    except AttributeError:
        pass  # Ignore special keys

def parse_keys(keys):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += force1                          
    if 'k' in keys:
        command[0:4] -= force1
    if 'j' in keys:
        command[[4,7]] += force2
        command[[5,6]] -= force2
    if 'l' in keys:
        command[[4,7]] -= force2
        command[[5,6]] += force2

    if 'w' in keys:
        command[4:8] += force1
    if 's' in keys:
        command[4:8] -= force1
    if 'a' in keys:
        command[[4,6]] += force1
    if 'd' in keys:
        command[[4,6]] -= force1
    return command

def parse_llm_output(llm_output):
    """
    Convert action strings returned by LLM to command array
    """
    command = np.zeros(8)
    if "ascend" in llm_output:
        command[0:4] += force1
    if "descend" in llm_output:
        command[0:4] -= force1
    if "move left" in llm_output:
        command[[4,7]] += force2
        command[[5,6]] -= force2
    if "move right" in llm_output:
        command[[4,7]] -= force2
        command[[5,6]] += force2
    if "move forward" in llm_output:
        command[4:8] += force1
    if "move backward" in llm_output:
        command[4:8] -= force1
    if "rotate left" in llm_output:
        command[[4,6]] += force1
    if "rotate right" in llm_output:
        command[[4,6]] -= force1
        command[[5,7]] += force1
    if "stop" in llm_output:
        command = np.zeros(8)
    return command

def parse_action_from_llm(llm_output):
    """
    Parse action commands from LLM output
    """
    action = ""
    if "ascend" in llm_output:
        action = "ascend"
    if "descend" in llm_output:
        action = "descend"
    if "move left" in llm_output:
        action = "move left"
    if "move right" in llm_output:
        action = "move right"
    if "move forward" in llm_output:
        action = "move forward"
    if "move backward" in llm_output:
        action = "move backward"
    if "rotate left" in llm_output:
        action = "rotate left"
    if "rotate right" in llm_output:
        action = "rotate right"
    if "stop" in llm_output:
        action = "stop"
    return action

def start_keyboard_listener():
    """
    Start the keyboard listener
    """
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()