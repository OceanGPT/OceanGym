import os
import yaml
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

# --- 新增：读取 YAML 配置文件 ---
# 获取当前文件所在的绝对路径，确保不管你在哪里运行脚本，都能正确定位到 config.yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
# 根据你的目录结构，config.yaml 放在 rov_task 的上一级（项目根目录）
yaml_path = os.path.join(current_dir, "../config.yaml")

with open(yaml_path, "r", encoding="utf-8") as f:
    yaml_config = yaml.safe_load(f)

# 提取路径
SEA_FLOOR_USD_PATH = yaml_config["paths"]["sea_floor_usd"]
ROBOT_USD_PATH = yaml_config["paths"]["robot_usd"]
# --------------------------------

@configclass
class ROVDockingEnvCfg(DirectRLEnvCfg):
    # Increase env_spacing to avoid large models overlapping each other
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=2000.0)

    observation_space = 13
    action_space = 6
    state_space = 13
    decimation = 4
    episode_length_s = 20.0 
    
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, 
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=1,
            gpu_max_rigid_contact_count=2**21, 
            gpu_max_rigid_patch_count=5 * 2**15,
            gpu_found_lost_pairs_capacity=2**21,
            enable_ccd=False,                  
        )
    )

    # 1. Static background: seabed scene
    sea_floor_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/StaticAssets/SeaFloor",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SEA_FLOOR_USD_PATH,  # <-- 已修改为从 YAML 读取的变量
            scale=(1.0, 1.0, 1.0),
        ),
    )

    # 2. Robot: scaled up by 400 times
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,      # <-- 已修改为从 YAML 读取的变量
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                         # Manually override mass to prevent automatic calculation into millions of tons
                
            ),
            scale=(200.0, 200.0, 200.0)
        ),
        # Initial position close to target point [176, 0, 521]
        init_state=ArticulationCfg.InitialStateCfg(pos=(190.0, 20.0, 1000.0)),
        actuators={}, 
    )
class ROVDockingEnv(DirectRLEnv):
    cfg: ROVDockingEnvCfg

    def __init__(self, cfg: ROVDockingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        import gymnasium as gym
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
        self.robot = self.scene.articulations["robot"]
        
        # Define target world coordinates
        self.target_pos_w = torch.tensor([176.0, 0.0, 521.0], device=self.device)
        
        # Thrust parameters tailored for 1000kg mass and 400x scaling
        self.k_f = 10000.0  # Base thrust gain
        self.k_m = 2000.0   # Base torque gain
        
        # Success threshold distance (considering the ROV itself is very large, 30-50m is a reasonable approach distance)
        self.success_threshold = 40.0

    def _setup_scene(self):
        # Generate the seabed
        self.cfg.sea_floor_cfg.spawn.func(self.cfg.sea_floor_cfg.prim_path, self.cfg.sea_floor_cfg.spawn)
        # Register the robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.1, 0.25, 0.35))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        u = torch.nan_to_num(self.actions) 
        num_envs = self.num_envs
        
        force_2d = torch.zeros((num_envs, 3), device=self.device)
        torque_2d = torch.zeros((num_envs, 3), device=self.device)

        # Thruster mapping logic
        force_2d[:, 0] = (u[:, 0] + u[:, 1] - u[:, 2] - u[:, 3]) * self.k_f * 0.707
        force_2d[:, 1] = (u[:, 0] - u[:, 1] + u[:, 2] - u[:, 3]) * self.k_f * 0.707
        torque_2d[:, 2] = (u[:, 0] - u[:, 1] - u[:, 2] + u[:, 3]) * self.k_m
        force_2d[:, 2] = (-u[:, 4] - u[:, 5]) * self.k_f
        torque_2d[:, 1] = (u[:, 4] - u[:, 5]) * self.k_m

        self.robot.set_external_force_and_torque(
            forces=force_2d.unsqueeze(1),
            torques=torque_2d.unsqueeze(1),
            body_ids=[0],
            env_ids=torch.arange(num_envs, device=self.device, dtype=torch.long),
            is_global=False
        )

    def _get_observations(self) -> dict:
        # Use world coordinates
        root_pos_w = self.robot.data.root_pos_w 
        target_rel_pos = self.target_pos_w - root_pos_w
        
        # Normalization: Because the model is huge, dividing the distance by 400 is more beneficial for network learning
        obs = torch.cat((
            target_rel_pos / 400.0,           # 3
            self.robot.data.root_quat_w,      # 4
            self.robot.data.root_lin_vel_w / 10.0, # 3
            self.robot.data.root_ang_vel_w    # 3
        ), dim=-1).to(torch.float32)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w 
        dist = torch.norm(self.target_pos_w - root_pos_w, dim=-1)
        
        # 1. Distance reward (scaled down to prevent value explosion)
        reward_dist = -1.0 * (dist / 400.0)
        # 2. Orientation reward (try to keep horizontal)
        reward_orientation = -0.5 * torch.sum(torch.square(self.robot.data.root_quat_w[:, 1:3]), dim=-1)
        # 3. Success reward
        reward_success = torch.where(dist < self.success_threshold, 200.0, torch.zeros_like(dist))
        
        return reward_dist + reward_orientation + reward_success

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos_w = self.robot.data.root_pos_w 
        dist = torch.norm(self.target_pos_w - root_pos_w, dim=-1)
        
        # Reset if successfully completed or too far from the target (e.g., more than 3000 meters)
        terminated = (dist < self.success_threshold) | (dist > 3000.0)
        truncated = self.episode_length_buf >= self.max_episode_length
        
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        root_state = self.robot.data.default_root_state[env_ids].clone()
        # The randomization range for the initial position is also expanded accordingly (e.g., randomized within a 100-meter range)
        root_state[:, :3] += (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 100.0
        
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)