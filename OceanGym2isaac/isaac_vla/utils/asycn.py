import torch
import numpy as np
import concurrent.futures
from threading import Lock

class AsyncAUVController:
    """
    Async AUV Controller Manager: Manages background LLM inference to prevent latency in the primary physics simulation loop.
    """
    def __init__(self, vla_client, speed_scale=30.0):
        self.client = vla_client
        self.speed_scale = speed_scale
        
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.lock = Lock()
        
        # Internal State
        self.future = None
        self.is_processing = False
        self.latest_command = "stop"
        
        # Velocity Mapping Table (Isaac Lab: +X is Forward, +Y is Left, +Z is Up)
        # Format: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
        self._velocity_map = {
            "forward":  np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "backward": np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "left":     np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "right":    np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            "ascend":   np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "descend":  np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
            "stop":     np.zeros(6)
        }

    def _parse_response(self, response: str) -> str:
        """Parses the text returned by the LLM and extracts control keywords"""
        res_lower = response.lower()
        if "move forward" in res_lower: return "forward"
        if "move backward" in res_lower: return "backward"
        if "move left" in res_lower: return "left"
        if "move right" in res_lower: return "right"
        if "ascend" in res_lower: return "ascend"
        if "descend" in res_lower: return "descend"
        return "stop"

    def step_request(self, img_paths: list, task_prompt: str):
        """
        Attempts to initiate a new inference request.If a request is 
        already being processed, it will automatically skip to avoid request buildup.
        """
        if not self.is_processing:
            self.is_processing = True
            # 将任务提交到线程池
            self.future = self.executor.submit(self.client.get_action, img_paths, task_prompt)

    def refresh_state(self):
        """
        Refreshes the background task status. Should be called every frame of the main loop.
        """
        if self.future and self.future.done():
            try:
                result = self.future.result()
                with self.lock:
                    self.latest_command = self._parse_response(result)
                    print(f"[AsyncHandler] New Command: {self.latest_command}")
            except Exception as e:
                print(f"[AsyncHandler] Inference Error: {e}")
            finally:
                self.is_processing = False
                self.future = None

    def get_velocity_tensor(self, device="cuda:0") -> torch.Tensor:
        """
        Retrieves the velocity Tensor to be executed. 
        Returns the previous persistent command even if no new instructions are received.
        """
        with self.lock:
            direction = self._velocity_map.get(self.latest_command, self._velocity_map["stop"])
            vel_np = direction * self.speed_scale
            
        return torch.tensor([vel_np], device=device, dtype=torch.float)

    @property
    def status(self):
        return "Thinking..." if self.is_processing else "Acting"