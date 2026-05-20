import argparse
import torch
import gymnasium as gym

# 1. Import Isaac Lab base components (Must be at the very top)
from isaaclab.app import AppLauncher

# Configure command line argument parsing
parser = argparse.ArgumentParser(description="Evaluate ROV Docking Policy")
parser.add_argument("--checkpoint", type=str, default="runs/ROV_Docking_Train/checkpoints/best_agent.pt", help="Path to the trained .pt file")
parser.add_argument("--task", type=str, default="Isaac-ROV-Docking-Direct-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize")

# Add arguments officially supported by Isaac Lab (e.g., --headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Start the simulation App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 2. Import remaining dependencies (Import after simulation_app starts)
# ---------------------------------------------------------
import rov_task 
from rov_task.rov_env import ROVDockingEnvCfg
# from train import Policy, Value  # Ensure train.py is in the same directory
from skrl.envs.wrappers.torch import IsaacLabWrapper
from rov_task.rov_env import ROVDockingEnvCfg
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.memories.torch import RandomMemory # Explicitly import
from skrl.utils import set_seed

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

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
    
def main():
    # --- Environment Preparation ---
    env_cfg = ROVDockingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # Randomization is usually turned off during evaluation to ensure deterministic observations
    # env_cfg.observations.policy.concatenate_history = False 
    
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabWrapper(raw_env)

    # --- Agent Configuration ---
    # This must remain consistent with the training configuration, otherwise network weights won't load properly
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    
    # If state normalization was enabled during training, it must be kept here
    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space, 
        "device": env.device
    }

    # --- Instantiate Models and Agent ---
    models = {
        "policy": Policy(env.observation_space, env.action_space, env.device),
        "value": Value(env.observation_space, env.action_space, env.device)
    }

    # Evaluation mode does not require memory
    agent = PPO(
        models=models, 
        memory=None, 
        cfg=agent_cfg, 
        observation_space=env.observation_space, 
        action_space=env.action_space, 
        device=env.device
    )

    # --- Load Weights ---
    print(f"\n[INFO] Loading model file: {args_cli.checkpoint}")
    try:
        agent.load("runs/ROV_Docking_Train/checkpoints/best_agent.pt")
        print("[INFO] Weights loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint. Please check the path or if the file is corrupted: {e}")
        return

    # Set to evaluation mode (turns off stochastic exploration, uses deterministic actions)
    agent.set_running_mode("test")

    # --- Run Inference Loop ---
    print("[INFO] Starting evaluation loop, press Ctrl+C to exit...")
    obs, _ = env.reset()
    
    try:
        while simulation_app.is_running():
            # Predict actions
            with torch.no_grad():
                # agent.act returns (actions, log_prob, ...) we take the first element
                actions = agent.act(obs, timestep=0, timesteps=0)[0]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(actions)
            
            # Simple real-time feedback
            if terminated.any() or truncated.any():
                print(f"[INFO] Episode finished, Current Reward: {reward[0].item():.4f}")
                # Isaac Lab Wrapper usually auto-resets, no manual reset needed
                
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user.")
    finally:
        # Clean up and close down
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()