import base64
import json
import requests
import os
import re
from datetime import datetime
from typing import List, Tuple

class VLAController:
    def __init__(self, api_key: str, model_name: str = "claude-haiku-4-5-20251001", api_base: str = "https://halogg.cn/v1"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base.rstrip('/')
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # log files
        self.log_dir = "vla_logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _save_response_to_file(self, raw_content: str):
        """将大模型的原始输出追加到日志文件中，并附带时间戳"""
        # 1. log name
        filename = os.path.join(self.log_dir, "all_responses.log")
        
        # 2.get time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        try:
          
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Timestamp: {current_time}\n")
                f.write(f"Content:\n{raw_content}\n")
                f.write(f"{'='*50}\n")
                
            print(f"[VLA] Response appended to {filename}")
        except Exception as e:
            print(f"[VLA Warning] Failed to save log: {e}")

    def get_action(self, image_paths: List[str], task_description: str) -> str:
        """
        获取动作指令。
        返回：提取出的指令字符串（例如 "move forward"）
        """
        print(f"[VLA] Processing {len(image_paths)} camera views...")
        
        view_labels = ["Front", "Back", "Left", "Right", "Up", "Down"]
        content_list = []
        
        # make up Prompt

        content_list.append({"type": "text", "text": task_description})

        for path, label in zip(image_paths, view_labels):
            try:
                b64 = self._encode_image(path)
                content_list.append({"type": "text", "text": f"--- {label} View ---"})
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}","detail": "low"}
                })
            except Exception as e:
                print(f"[VLA Warning] Failed to load {label} image: {e}")

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content_list}],
            "temperature": 0.2
            
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions", 
                headers=self.headers, 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            raw_content = result['choices'][0]['message']['content'].strip()
            
            # 1. save log to files
            self._save_response_to_file(raw_content)
            
            # 2. extract final  command
            # format: $$ Analysis @@@ reason: $$ Instruction
         
            if "$$" in raw_content:
                # find position of final $$
                last_marker_index = raw_content.rfind("$$")
                # extract command
                instruction = raw_content[last_marker_index + 2:].strip()
                # eliminate interference
                instruction = instruction.replace("，", "").strip()
            else:
                # if not found, return full command
                instruction = raw_content

            print(f"[VLA Success] Extracted Instruction: {instruction}")
            return instruction

        except Exception as e:
            print(f"[VLA Error] Inference failed: {e}")
            return "stop" # issue stop when error occur


