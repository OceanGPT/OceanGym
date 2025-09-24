import argparse
import datetime
import os
import random
import sys
import time
import yaml
from typing import List, Dict, Any

import cv2
import holoocean
import matplotlib.pyplot as plt
import numpy as np
from pynput import keyboard

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

from utils import *


# Global variable area
pressed_keys = list()
z_key_flag = {"pressed": False, "needs_saving": False}  # Use a dictionary to avoid scope issues


def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

        if key.char == 'z':
            z_key_flag["pressed"] = True
            z_key_flag["needs_saving"] = True
            print("Captured 'Z' key press event, marked save task")


def on_release(key):
    global pressed_keys
    try:
        if hasattr(key, 'char'):
            pressed_keys.remove(key.char)
            # Clear pressed flag when 'Z' key is released, but keep needs_saving flag for main loop to handle
            if key.char == 'z':
                z_key_flag["pressed"] = False
    except:
        pass


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


def parse_keys(keys, val_spin, val_verti):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val_verti
    if 'k' in keys:
        command[0:4] -= val_verti
    if 'j' in keys:
        command[[4,7]] += val_spin
        command[[5,6]] -= val_spin
    if 'l' in keys:
        command[[4,7]] -= val_spin
        command[[5,6]] += val_spin

    if 'w' in keys:
        command[4:8] += val_verti
    if 's' in keys:
        command[4:8] -= val_verti
    if 'a' in keys:
        command[[4,6]] += val_verti
        command[[5,7]] -= val_verti
    if 'd' in keys:
        command[[4,6]] -= val_verti
        command[[5,7]] += val_verti

    return command


def save_image(data, prefix, base_dir, task_name=""):
    """
    Save image data (numpy array or matplotlib Figure) to disk with timestamped filename.
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    task_name = task_name or date_str
    time_str = now.strftime("%H-%M-%S")
    filename = f"{prefix}--{time_str}.png"
    full_dir = os.path.join(base_dir, task_name)
    os.makedirs(full_dir, exist_ok=True)
    full_path = os.path.join(full_dir, filename)

    if data is None:
        print(f"Error: Cannot save {prefix}, data is empty")
        return False

    try:
        if isinstance(data, plt.Figure):
            # Ensure the figure is visible and fully rendered
            plt.figure(data.number)
            data.canvas.draw()
            data.savefig(full_path, bbox_inches='tight', pad_inches=0)
            print(f"Image saved: {full_path}")
            return True

        if data.ndim == 3 and data.shape[-1] == 4:
            data = data[:, :, :3]

        if data.dtype != np.uint8:
            data = np.clip(data, 0, None)
            if data.max() > 1.0 + 1e-4:
                data = (data / 255.0).astype(np.float32)
            data = (data * 255).astype(np.uint8)

        success = cv2.imwrite(full_path, data)
        if success:
            print(f"Image saved: {full_path}")
            return True
        else:
            print(f"Error: OpenCV cannot write image: {full_path}")
            return False
    except Exception as e:
        print(f"Error occurred while saving image: {e}")
        return False


def save_image_async(data, prefix, base_dir, task_name):
    """Submit save_image task asynchronously"""
    executor.submit(save_image, data, prefix, base_dir, task_name)


def check_zones(yaml_path, location):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        distinations: List[Dict[str, Any]] = yaml.safe_load(f)

    hited_names: List[str] = []

    for distination in distinations:
        disti_name = distination['name']
        disti_type = distination['type']
        disti_center = np.array(distination['center'], dtype=float)
        disti_r = float(distination['r'])

        if disti_type == "sphere":
            hit = np.linalg.norm(location - disti_center) <= disti_r
        elif disti_type == "cylinder":
            hit = np.linalg.norm(location[:2] - disti_center) <= disti_r
        else:
            continue

        if hit:
            hited_names.append(disti_name)

    return hited_names


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == "__main__":
    log_path = os.path.join(os.path.dirname(__file__), "run_noSonar.log")
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    RGB_CAM_LIST = ['FrontCamera', 'BackCamera', 'LeftCamera', 'RightCamera', 'UpCamera', 'DownCamera']

    GROUP_IDX = 1  # Image group index, start from this number

    parser = argparse.ArgumentParser(description='Start Ocean Simulation Map')
    parser.add_argument(
        '--scenario',
        type=str,
        default='Dam-Hovering',
        help='holoocean Scenario name',
    )
    parser.add_argument(
        '--img_capture',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'data'),
        help='Root directory to save captured images',
    )
    parser.add_argument(
        '--task_name',
        type=str,
        default='test',
        help='Sub directory to save captured images',
    )
    parser.add_argument(
        '--distinations',
        type=str,
        default='config/distinations.yaml',
        help='YAML file path for destination zones',
    )
    parser.add_argument(
        '--val_spin',
        type=float,
        default=1,
        help='Spin value for rotation control',
    )
    parser.add_argument(
        '--val_verti',
        type=float,
        default=100,
        help='Vertical thrust value for movement control',
    )
    parser.add_argument(
        '--rgbcamera',
        type=str,
        default='all',
        choices=RGB_CAM_LIST + ['all'],
        help='Select which camera feed to display and save',
    )
    args = parser.parse_args()

    scenario = args.scenario
    img_capture_output_path = args.img_capture

    config = holoocean.packagemanager.get_scenario(scenario)

    config = config['agents'][0]['sensors'][-1]['configuration']

    # sonar part
    try:
        azi = config['Azimuth']
        minR = config['RangeMin']
        maxR = config['RangeMax']
        binsR = config['RangeBins']
        binsA = config['AzimuthBins']
        plt.ion()
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8,5))
        ax.set_theta_zero_location("N")
        ax.set_thetamin(-azi/2)
        ax.set_thetamax(azi/2)
        theta = np.linspace(-azi/2, azi/2, binsA)*np.pi/180
        r = np.linspace(minR, maxR, binsR)
        T, R = np.meshgrid(theta, r)
        z = np.zeros_like(T)
        plt.grid(False)
        plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
    except:
        print("Sonar configuration not found or invalid. Skipping sonar visualization.")

    z_pressed_last_frame = False
    with holoocean.make(scenario, scenario_cfg=holoocean.packagemanager.get_scenario(scenario), start_world=False) as env:
        while True:
            if 'q' in pressed_keys:
                break

            command = parse_keys(keys=pressed_keys, val_spin=args.val_spin, val_verti=args.val_verti)
            env.act("auv0", command)
            state = env.tick()

            # Arrival Check
            current_location = state["LocationSensor"]
            current_rotation = state["RotationSensor"]

            camera_to_use = args.rgbcamera

            if z_key_flag["needs_saving"]:
                print(f"Saving image group G{GROUP_IDX}...")

                # First, save a copy of the current state to ensure the same moment's data is saved
                current_state = state.copy()

                # Lock the current group number and prepare to save
                current_group = GROUP_IDX
                has_saved = False  # Used to track if any images were successfully saved

                # Save camera images
                for cam in RGB_CAM_LIST:
                    if cam in current_state and (camera_to_use == 'all' or camera_to_use == cam):
                        print(f"Saving camera image G{current_group}_{cam}")
                        try:
                            # Save camera image synchronously
                            success = save_image(current_state[cam],
                                               f"G{current_group}_{cam}",
                                               args.img_capture,
                                               args.task_name)
                            if success:
                                has_saved = True
                        except Exception as e:
                            print(f"Saving camera image {cam} failed: {e}")

                if has_saved:
                    GROUP_IDX += 1  # only increment group number if any image was saved
                    print(f"✓ Image group G{current_group} saved")
                    print(f"Location: {current_location}, Rotation: {current_rotation}")
                else:
                    print("⚠ No images were saved!")

                z_key_flag["needs_saving"] = False
                print("===== Save task completed, waiting for next press of Z key =====")

            # RGB Camera * n
            for cam_name in RGB_CAM_LIST:
                if cam_name in state and (camera_to_use == 'all' or camera_to_use == cam_name):
                    pixels = state[cam_name]
                    cv2.namedWindow(f"{cam_name} Output")
                    cv2.imshow(f"{cam_name} Output", pixels[:, :, 0:3])
                    cv2.waitKey(1)

    executor.shutdown(wait=True)
    cv2.destroyAllWindows()

    plt.ioff()
    plt.show()
