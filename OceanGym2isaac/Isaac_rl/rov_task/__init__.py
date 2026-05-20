# rov_task/__init__.py
import gymnasium as gym
from .rov_env import ROVDockingEnv, ROVDockingEnvCfg

gym.register(
    id="Isaac-ROV-Docking-Direct-v0",
    entry_point="rov_task.rov_env:ROVDockingEnv",
    disable_env_checker=True,
    kwargs={
        "cfg_entry_point": ROVDockingEnvCfg,  # 确保这里传入的是类对象
    },
)