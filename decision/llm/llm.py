import yaml
import logging
from datetime import datetime
import re
import numpy as np
import os
import sys
import base64
import io
from PIL import Image
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from decision.utils.config_process import load_config
# Add current directory to path

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

def get_llm_config():
    """Load LLM config for API mode only"""
    try:
        config = load_config()
        print("🌐 API mode")
        return config
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}, using default API config")
        return None

CONFIG = get_llm_config()
_api_client = None

def _get_api_client():
    """Lazy initialization of API client (API mode only)"""
    global _api_client
    if _api_client is None:
        from openai import OpenAI
        if CONFIG:
            api_config = CONFIG.get("llm", {}).get("api", {})
            api_key = api_config.get("api_key")
            base_url = api_config.get("base_url")
            model_name = api_config.get("model")
            if not api_key:
                raise ValueError("API key must be provided in config['llm']['api']['api_key']")
        else:
            raise ValueError("No config loaded, cannot initialize API client without API key")
        _api_client = {
            'client': OpenAI(api_key=api_key, base_url=base_url),
            'model': model_name
        }
        print("✅ API client initialized successfully")
    return _api_client

def ask_llm(prompt, b64_image_lst, max_retries=3):
    """
    Call large language model interface (API mode only).
    Always return a string.
    """
    print("🌐 API mode")
    return ask_llm_api(prompt, b64_image_lst, max_retries)

def ask_llm_api(prompt, b64_image_lst, max_retries=3):
    """API mode LLM call. Always return a string."""
    api_info = _get_api_client()
    client = api_info['client']
    model_name = api_info['model']

    for attempt in range(max_retries + 1):
        try:
            content_lst = [{"type": "text", "text": prompt}]
            for image in b64_image_lst:
                content_lst.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"}
                })
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content_lst}],
                max_tokens=1024
            )

            if isinstance(response, str):
                return response.strip()
            if hasattr(response, "choices"):
                return response.choices[0].message.content.strip()
            return str(response)
        except Exception as e:
            if attempt < max_retries:
                logging.warning(f"API call failed, retrying in {2**attempt} seconds: {e}")
                time.sleep(2 ** attempt)
            else:
                raise e

def found_target(target_info):
    """Check if target object is found"""
    return "@@@" in target_info

def get_llm_info(config):
    """Get LLM information for compatibility (API mode only)"""
    try:
        print(f"🔍 Debug: config structure = {config.keys()}")
        llm_config = config.get("llm", {})
        api_config = llm_config.get("api", {})
        if api_config:
            model_name = api_config.get("model", "gpt-4o-mini")
        else:
            model_name = config.get("defaults", {}).get("llm", {}).get("model", "gpt-4o-mini")
        log_name = f"{model_name}-api"
        print(f"✅ Returning API mode info: api, {model_name}, {log_name}")
        return "api", model_name, log_name
    except Exception as e:
        print(f"❌ Failed to get LLM info: {e}")
        import traceback
        traceback.print_exc()
        return "api", "gpt-4o-mini", "gpt-4o-mini-api"

def get_llm_status():
    """Get current LLM status (API mode only)"""
    status = {
        "mode": "api",
        "use_local": False,
        "config_loaded": CONFIG is not None
    }
    return status

def test_llm_integration():
    """Test LLM integration (API mode only)"""
    try:
        print("="*60)
        print(f"🧪 Testing LLM integration")
        status = get_llm_status()
        print(f"📊 Current status: {status}")
        test_prompt = "Hello, please answer briefly: What is 1+1?"
        print(f"👤 Test prompt: {test_prompt}")
        response = ask_llm(test_prompt, [])
        print(f"✅ Test successful!")
        print(f"🤖 Response: {response}")
        test_response = "This is a test response @@@"
        found = found_target(test_response)
        print(f"🔍 Target detection test: {'✅ Success' if found else '❌ Failed'}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"🔧 LLM Mode: API")
    print(f"CONFIG: {CONFIG}")
    test_llm_integration()