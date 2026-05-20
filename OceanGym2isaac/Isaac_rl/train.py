import argparse
import torch
import gymnasium as gym
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train ROV with skrl")
parser.add_argument("--task", type=str, default="Isaac-ROV-Docking-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import rov_task 
from rov_task.rov_env import ROVDockingEnvCfg
from skrl.envs.wrappers.torch import IsaacLabWrapper
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.memories.torch import RandomMemory # Explicitly import
from skrl.utils import set_seed

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, reduction="sum")
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64), torch.nn.ELU(),
            torch.nn.Linear(64, self.num_actions)
        )
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))
    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64), torch.nn.ELU(),
            torch.nn.Linear(64, 1)
        )
    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

from skrl.resources.preprocessors.torch import RunningStandardScaler

def main():
    set_seed(42)
    env_cfg = ROVDockingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabWrapper(raw_env)

    # --- 1. Configure PPO Agent ---
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    
    # State normalization: must be enabled to resolve your previous issue with large observation values
    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space, 
        "device": env.device,
        "epsilon": 1e-8,
    }
    
    # Training cadence configuration
    # Note: in skrl, the configuration key is usually "rollouts", please confirm if your version uses "rollout_steps" instead
    rollout_steps = 32
    agent_cfg["rollouts"] = rollout_steps 
    agent_cfg["learning_epochs"] = 6
    agent_cfg["mini_batches"] = 4
    agent_cfg["learning_rate"] = 3e-4  # Robust learning rate
    agent_cfg["grad_norm_clip"] = 0.5   # Strongly prevent gradient explosion

    agent_cfg["experiment"]["directory"] = "runs"
    agent_cfg["experiment"]["experiment_name"] = "ROV_Docking_Train"
    agent_cfg["experiment"]["write_interval"] = 100       # Write to TensorBoard every 100 steps
    agent_cfg["experiment"]["checkpoint_interval"] = 500  # Save a model checkpoint every 500 steps
    
    # --- 2. Instantiate Models ---
    models = {
        "policy": Policy(env.observation_space, env.action_space, env.device),
        "value": Value(env.observation_space, env.action_space, env.device)
    }
    
    # Memory size must be equal to rollout_steps
    memory = RandomMemory(memory_size=rollout_steps, num_envs=env.num_envs, device=env.device)

    # --- 3. Instantiate Agent ---
    # Do not set _current_states manually here; let the Trainer manage it instead
    agent = PPO(
        models=models, 
        memory=memory, 
        cfg=agent_cfg, 
        observation_space=env.observation_space, 
        action_space=env.action_space, 
        device=env.device
    )

    # --- 4. Configure and Start Trainer ---
    # SequentialTrainer automatically calls env.reset() and processes the initial observations
    trainer_cfg = {
        "timesteps": 100000, 
        "headless": args_cli.headless,
    }
    
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    
    print("[INFO]: Starting ROV Training...")
    trainer.train()

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()