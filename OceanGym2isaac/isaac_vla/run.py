import argparse
import torch
import numpy as np
import warp as wp
import random
import os
import json
import yaml  

# 1. basic AppLauncher asset
from isaaclab.app import AppLauncher

# --- Argument Parsing Section ---
parser = argparse.ArgumentParser(description="Isaac Lab AUV VLA Closed-Loop Control System - Fully Automated Version")
parser.add_argument("--mode", type=str, default="vla", choices=["vla", "keyboard"], help="Control mode: 'vla' for AI, 'keyboard' for manual")
parser.add_argument("--config", type=str, default="auv_config.yaml", help="Path to the config file") # Added parameter
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- Load YAML Configuration File ---
with open(args_cli.config, 'r') as f:
    cfg = yaml.safe_load(f)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac Lab Related Libraries ---
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg  
import isaaclab.utils.math as math_utils  # Added: for quaternion to rotation matrix conversion
import omni.ui as ui
import carb.input
import omni.appwindow
import carb

# Import custom utility classes
from utils.camera import UnderwaterCameraManager, UnderwaterScene
from utils.llm import VLAController
from utils.asycn import AsyncAUVController
from utils.prompt.dock import formatted_prompt

def enable_translucency():
    # Get settings interface
    settings = carb.settings.get_settings()
    
    # Enable translucency setting under RTX Real-Time mode
    # Note: If using Path Tracing, the path will be different
    settings.set("/rtx/translucency/enabled", True)
    
    # Optional: Further optimize underwater translucency parameters (e.g., refraction bounces)
    settings.set("/rtx/translucency/maxRefractionBounces", 4)
    print("[INFO] Translucency rendering option enabled automatically")

# --- Original Utility Functions (Logic fully preserved) ---
def get_single_random_auv_pos():
    x_range = (-500.0, 500.0)
    y_range = (-500.0, 500.0)
    z_range = (1300.0, 1800.0)
    x = float(random.uniform(*x_range))
    y = float(random.uniform(*y_range))
    z = float(random.uniform(*z_range))
    return (x, y, z)

def get_speed(a):
    if a < 500.0:
        speed = 1.0
    elif a >= 500 and a < 1800:
        ratio = (a - 500.0) / (1800.0 - 500.0)
        speed = 1.0 + ratio * (3.0 - 1.0)
    else:
        speed = 3.0
    return speed

wp.init()

@configclass
class OceanSceneCfg(InteractiveSceneCfg):
    main_light = AssetBaseCfg(
        prim_path="/World/MainLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=2000.0,
            color=(1.0, 1.0, 1.0),
        ),
    )

    ambient_light = AssetBaseCfg(
        prim_path="/World/AmbientLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=300.0,
            color=(0.8, 0.9, 1.0),
        ),
    )
    ocean_floor = AssetBaseCfg(
        prim_path="/World/OceanFloor",
        spawn=sim_utils.UsdFileCfg(
            usd_path=cfg['paths']['floor_usd'], # Modified to read from config file
            scale=(1.0, 1.0, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    )

    auv = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/AUV",
        spawn=sim_utils.UsdFileCfg(
            usd_path=cfg['paths']['auv_usd'], # Modified to read from config file
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_depenetration_velocity=1.0,
                disable_gravity=True, 
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0), 
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True,),
            func=sim_utils.spawners.from_files.spawn_from_usd, 
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=get_single_random_auv_pos(),
            rot=(1.0, 0, 0, 0),
        ),
    )

    # Camera configurations remain completely unchanged...
    cam_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_front",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(50,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(1, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),convention='world')
    )
    cam_back = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_back",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(50,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(-1, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0),convention='world')
    )
    cam_left = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_left",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(25,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 1, 0.0), rot=(0.707, 0.0, 0.0, 0.707),convention='world')
    )
    cam_right = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_right",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(25,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, -1, 0.0), rot=(0.707, 0.0, 0.0, -0.707),convention='world')
    )
    cam_up = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_up",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(25,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 1), rot=(0.707, 0.0, -0.707, 0.0),convention='world')
    )
    cam_down = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_down",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(25,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, -100), rot=(0.707, 0.0, 0.707, 0.0),convention='world')
    )
    cam_follow = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_follow",
        update_period=0, height=960, width=1280, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(-500, 0.0, 500), rot=(0.92388, 0.0, 0.38268, 0.0), convention='world')
    )

def main():
    # 1. scene initial
    sim_context = SimulationContext(sim_utils.SimulationCfg(dt=0.02, device="cuda:0"))
    scene = InteractiveScene(OceanSceneCfg(num_envs=1, env_spacing=0.0))
    
    # 2. sensors and UI
    camera_names = ["cam_front", "cam_back", "cam_left", "cam_right", "cam_up", "cam_down"]
    camera_sensors = {name: scene[name] for name in camera_names}
    auv_robot: RigidObject = scene["auv"]
    
    window = ui.Window(f"AUV View - Mode: {args_cli.mode}", width=1280, height=720)
    provider = ui.ByteImageProvider()
    with window.frame:
        ui.ImageWithProvider(provider)

    # 3. tools initialization
    cam_manager = UnderwaterCameraManager(camera_sensors)
    monitor_cam_manager = UnderwaterScene(scene["cam_follow"], device="cuda:0")
    
    # --- Automation and Logging Logic Variables ---
    trajectory_log = []           
    MAX_STEPS = 9000             
    STOP_THRESHOLD = 5            
    consecutive_stop_count = 0 
    
    # --- Controller Branch Initialization ---
    if args_cli.mode == "vla":
        # Modified to read API Key from configuration file
        vla_client = VLAController(api_key=cfg['vla']['api_key'],model_name=cfg['vla']['model_name'],api_base=cfg['vla']['api_base'])
        async_ctrl = AsyncAUVController(vla_client, speed_scale=10.0)
    else:
        kb_cfg = Se2KeyboardCfg(v_x_sensitivity=400.0, v_y_sensitivity=400.0)
        keyboard = Se2Keyboard(kb_cfg)
        _input = carb.input.acquire_input_interface()
        _kb_device = omni.appwindow.get_default_app_window().get_keyboard()
        print("[INFO] Manual Control: Arrow keys (Body-relative direction), PageUp/PageDown (Altitude), Q/E (Rotate Left/Right)")

    sim_context.reset()
    enable_translucency()
    current_vel = torch.zeros((1, 6), device="cuda:0")
    step_count = 0
    
    print("[INFO] Simulation starting...")

    while simulation_app.is_running():
        if sim_context.is_playing():
            if step_count >= MAX_STEPS:
                print(f"[TERMINATE] Maximum step limit reached ({MAX_STEPS} steps), stopping automatically.")
                break

            auv_robot.update(dt=0.02)
            auv_pos_tensor = auv_robot.data.root_pos_w[0]
            auv_pos_np = auv_pos_tensor.detach().cpu().numpy()
            trajectory_log.append(auv_pos_np.tolist()) 
            
            current_z = auv_pos_np[2]
            
            if args_cli.mode == "keyboard":
                kb_data = keyboard.advance()
                vz = 0.0
                wz = 0.0  
                
                # Altitude Control
                if _input.get_keyboard_value(_kb_device, carb.input.KeyboardInput.PAGE_UP):
                    vz = 400.0
                elif _input.get_keyboard_value(_kb_device, carb.input.KeyboardInput.PAGE_DOWN):
                    vz = -400.0
                
                # Rotation Control: Q for Left Yaw, E for Right Yaw
                if _input.get_keyboard_value(_kb_device, carb.input.KeyboardInput.Q):
                    wz = 2.0  
                elif _input.get_keyboard_value(_kb_device, carb.input.KeyboardInput.E):
                    wz = -2.0
                
                # 1. Assemble desired velocity in Body Frame
                # kb_data[0] represents body forward/backward, kb_data[1] represents body left/right
                body_vel_lin = torch.tensor([[kb_data[0], kb_data[1], vz]], device="cuda:0") 

                # 2. Get AUV current quaternion orientation in World Frame [w, x, y, z]
                auv_quat_w = auv_robot.data.root_quat_w[0:1] 

                # 3. Rotate linear velocity vector from Body Frame to World Frame
                world_vel_lin = math_utils.quat_rotate(auv_quat_w, body_vel_lin)

                # 4. Write converted World Frame linear velocity and angular velocity into control tensor
                new_vel = torch.zeros((1, 6), device="cuda:0")
                new_vel[0, 0:3] = world_vel_lin[0] # Transformed World Frame linear velocity vx, vy, vz
                new_vel[0, 5] = wz                 # Rotation angular velocity wz
                current_vel = new_vel
            
            auv_robot.write_root_velocity_to_sim(current_vel * get_speed(current_z))

            processed_tensor = monitor_cam_manager.process_frame()
            if processed_tensor is not None:
                provider.set_bytes_data_from_gpu(
                    processed_tensor.data_ptr(), 
                    (monitor_cam_manager.width, monitor_cam_manager.height)
                )
            
        sim_context.step(render=True)
        
        if sim_context.is_playing() and args_cli.mode == "vla":
            if step_count % 100 == 0:
                img_paths = cam_manager.capture_all_and_process()
                if img_paths:
                    task = """
**[Your Task]**
As the Control Expert, based on your analysis of the provided real-time images, execute your decision cycle according to the following strict protocol:
1.  **PRIMARY ANALYSIS - TARGET ACQUISITION:** Scrutinize the six (6) non-target, real-time camera feeds provided. Is the target object visible in any of them?
2.  **DECISION & ACTION:**
    -   **IF TARGET IS VISIBLE:** You must immediately initiate the 'Target ' protocol. Report using the "$$...$$" format and issue commands to approach and position over the target.
            command is one of [ `ascend`, `descend`, `move left`, `move right`, `move forward`, `move backward`, `stop`]
"""
                    prompt = formatted_prompt + "\n" + task
                    async_ctrl.step_request(img_paths, prompt)
                    async_ctrl.refresh_state()
                    
                    current_vel = async_ctrl.get_velocity_tensor()
                    
                    if torch.all(current_vel == 0):
                        consecutive_stop_count += 1
                        print(f"[VLA] Consecutive STOP command received: count={consecutive_stop_count}")
                    else:
                        consecutive_stop_count = 0 
                    
                    if consecutive_stop_count >= STOP_THRESHOLD:
                        print(f"[TERMINATE] VLA sent {STOP_THRESHOLD} consecutive STOP signals. Task completed.")
                        break
       
            step_count += 1
            
    # Trajectory saving logic remains unchanged...
    parent_dir = "experiment"
    sub_dir = "llm"
    target_folder = os.path.join(parent_dir, sub_dir)
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    base_name = "auv_trajectory"
    extension = ".json"
    counter = 1
    while os.path.exists(os.path.join(target_folder, f"{base_name}_{counter}{extension}")):
        counter += 1
    
    log_file_name = f"{base_name}_{counter}{extension}"
    log_path = os.path.join(target_folder, log_file_name)

    try:
        with open(log_path, "w") as f:
            json.dump(trajectory_log, f, indent=4)
        print(f"\n[SUCCESS] Experimental data exported to: {log_path}")
    except Exception as e:
        print(f"[ERROR] Error writing to log file: {e}")
        
    simulation_app.close()

if __name__ == "__main__":
    main()
