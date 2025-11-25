"""
'mvsex' means 'multi-view with few sonar examples'
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from openai import OpenAI
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

from utils import *


# If you use Windows, please change the path accordingly.
ship_img = "asset/perception/sonar_img/ship.png"
plane_img = "asset/perception/sonar_img/plane.png"
normal_pipeline_img = "asset/perception/sonar_img/normal-pipeline.png"
damaged_pipeline_img = "asset/perception/sonar_img/damaged-pipeline.png"
tower_img = "asset/perception/sonar_img/tower.png"
container_box_img = "asset/perception/sonar_img/container-box.png"

ship_b64 = encode_image_to_base64(ship_img)
plane_b64 = encode_image_to_base64(plane_img)
normal_pipeline_b64 = encode_image_to_base64(normal_pipeline_img)
damaged_pipeline_b64 = encode_image_to_base64(damaged_pipeline_img)
tower_b64 = encode_image_to_base64(tower_img)
container_box_b64 = encode_image_to_base64(container_box_img)


def call_OpenAI(
        model_name_or_path: str,
        api_key: str,
        base_url: str,
        image_input: str,
        options: List[str],
        prompt: str,
        prompt_sonar: str,
    ) -> Tuple[List[str], str, List[Dict[str, Any]]]:

    img_b64 = encode_image_to_base64(image_input)
    img_b64_sonar = encode_image_to_base64(image_input.replace("FrontCamera", "sonar"))

    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{normal_pipeline_b64}"},
                {"type": "input_text", "text": "This sonar image example of is **normal-pipeline**."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{damaged_pipeline_b64}"},
                {"type": "input_text", "text": "This sonar image example of is **damaged-pipeline**."},
                {"type": "input_image",  "image_url": f"data:image/png;base64,{ship_b64}"},
                {"type": "input_text", "text": "This sonar image example of is **ship**."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{plane_b64}"},
                {"type": "input_text", "text": "This sonar image example of is **plane**."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{tower_b64}"},
                {"type": "input_text", "text": "This sonar image example of is **tower**."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{container_box_b64}"},
                {"type": "input_text", "text": "This sonar image example of is **container-box**."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64_sonar}"},
                {"type": "input_text", "text": prompt_sonar},
                {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                {"type": "input_text", "text": prompt}
            ]
        }
    )

    client = client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    resp = client.responses.create(
        model=model_name_or_path,
        input=messages,
        temperature=0,  # To ensure deterministic output
        top_p=1, # To ensure deterministic output
    )
    raw_output = resp.output_text.strip()
    list_output = normalize_model_output(raw_output, options)

    messages.append({"role": "assistant", "content": resp.output_text})
    return list_output, raw_output, messages


def call_Gemma(
        device: str,
        model_name_or_path: str,
        image_input: str,
        options: List[str],
        prompt: str,
        prompt_sonar: str,
    ) -> Tuple[List[str], str, List[Dict[str, Any]]]:
    from transformers import Gemma3ForConditionalGeneration

    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).to(torch.device(device)).eval()
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": normal_pipeline_img},
                {"type": "text", "text": "This sonar image example of is **normal-pipeline**."},
                {"type": "image", "image": damaged_pipeline_img},
                {"type": "text", "text": "This sonar image example of is **damaged-pipeline**."},
                {"type": "image", "image": ship_img},
                {"type": "text", "text": "This sonar image example of is **ship**."},
                {"type": "image", "image": plane_img},
                {"type": "text", "text": "This sonar image example of is **plane**."},
                {"type": "image", "image": tower_img},
                {"type": "text", "text": "This sonar image example of is **tower**."},
                {"type": "image", "image": container_box_img},
                {"type": "text", "text": "This sonar image example of is **container-box**."},
                {"type": "image", "image": image_input.replace("FrontCamera", "sonar")},
                {"type": "text", "text": prompt_sonar},
                {"type": "image", "image": image_input},
                {"type": "text", "text": prompt}
            ]
        }
    )

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,     # To ensure deterministic output
            temperature=0.0,     # To ensure deterministic output
            top_p=1.0, # To ensure deterministic output
            top_k=50, # To ensure deterministic output
        )
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

    raw_output = decoded.strip()
    list_output = normalize_model_output(raw_output, options)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": decoded}]})

    import gc
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    return list_output, raw_output, messages


def call_Qwen(
        device: str,
        model_name_or_path: str,
        image_input: str,
        options: List[str],
        prompt: str,
        prompt_sonar: str,
    ) -> Tuple[List[str], str, List[Dict[str, Any]]]:
    from transformers import Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    img_b64_sonar = encode_image_to_base64(image_input.replace("FrontCamera", "sonar"))
    img_b64 = encode_image_to_base64(image_input)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path, torch_dtype="auto", device_map="auto"
    ).to(torch.device(device))
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
    )

    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image_url": f"data:image;base64,{normal_pipeline_b64}"},
                {"type": "text", "text": "This sonar image example of is **normal-pipeline**."},
                {"type": "image", "image_url": f"data:image;base64,{damaged_pipeline_b64}"},
                {"type": "text", "text": "This sonar image example of is **damaged-pipeline**."},
                {"type": "image", "image_url": f"data:image;base64,{ship_b64}"},
                {"type": "text", "text": "This sonar image example of is **ship**."},
                {"type": "image", "image_url": f"data:image;base64,{plane_b64}"},
                {"type": "text", "text": "This sonar image example of is **plane**."},
                {"type": "image", "image_url": f"data:image;base64,{tower_b64}"},
                {"type": "text", "text": "This sonar image example of is **tower**."},
                {"type": "image", "image_url": f"data:image;base64,{container_box_b64}"},
                {"type": "text", "text": "This sonar image example of is **container-box**."},
                {"type": "image", "image_url": f"data:image;base64,{img_b64_sonar}"},
                {"type": "text", "text": prompt_sonar},
                {"type": "image", "image_url": f"data:image;base64,{img_b64}"},
                {"type": "text", "text": prompt}
            ]
        }
    )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(torch.device(device))

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,   # To ensure deterministic output
        temperature=0.0,   # To ensure deterministic output
        top_p=1.0, # To ensure deterministic output
        top_k=50, # To ensure deterministic output
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text)
    raw_output = output_text[0].strip()
    list_output = normalize_model_output(raw_output, options)
    messages.append({"role": "assistant", "content": output_text})

    import gc
    del model, processor, image_inputs, video_inputs, inputs, generated_ids, generated_ids_trimmed, output_text
    torch.cuda.empty_cache()
    gc.collect()
    return list_output, raw_output, messages


def call_BMB(
        device: str,
        model_name_or_path: str,
        image_input: str,
        options: List[str],
        prompt: str,
        prompt_sonar: str,
    ) -> Tuple[List[str], str, List[Dict[str, Any]]]:

    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        attn_implementation='sdpa', # sdpa or flash_attention_2, no eager
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().to(torch.device(device))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )

    image_sonar = Image.open(image_input.replace("FrontCamera", "sonar")).convert('RGB')
    image = Image.open(image_input).convert('RGB')

    messages = []
    messages.append(
        {'role': 'user', 'content': [
            normal_pipeline_img, "This sonar image example of is **normal-pipeline**.",
            damaged_pipeline_img, "This sonar image example of is **damaged-pipeline**.",
            ship_img, "This sonar image example of is **ship**.",
            plane_img, "This sonar image example of is **plane**.",
            tower_img, "This sonar image example of is **tower**.",
            container_box_img, "This sonar image example of is **container-box**.",
            image_sonar, prompt_sonar,
            image, prompt
            ]
        }
    )

    answer = model.chat(
        msgs=messages,
        tokenizer=tokenizer,
        enable_thinking=False,
        stream=False,
    )

    list_output = normalize_model_output(answer, options)
    raw_output = answer

    messages.clear() # avoid storing large images in the log
    messages.append({'role': 'user', 'content': [image_input.replace("FrontCamera", "sonar"), prompt_sonar, image_input, prompt]})
    messages.append({"role": "assistant", "content": [answer]})

    import gc
    del model, tokenizer, image
    torch.cuda.empty_cache()
    gc.collect()
    return list_output, raw_output, messages


if __name__ == "__main__":
    # Golbal Settings:

    set_seed(42)  # To ensure deterministic output

    DEVICE = "cuda:0"

    MODELS_TEMPLATE = ['OpenAI', 'Qwen', 'BMB', 'Gemma']

    OPTS = ["plane", "ship", "normal-pipeline", "damaged-pipeline", "tower", "container-box"]

    PROMPT_SONAR = f"This sonar image can be used as a reference to assist in identifying the next color image."

# Old Prompt
#     PROMPT00 = f"""You are an assistant that analyzes an image and checks which of the following options appear in it. Before that, I have already provide you a sonar image for reference only.

# Options:
# {OPTS}

# Instructions:
# - Only when you find it difficult to recognize the color image, I suggest you refer to the previous sonar image together.
# - Carefully examine the image, even the corners.
# - You can choose single or multiple options, if none of the options appear, just return an empty Python list.
# - For multiple-choice questions, no points will be awarded for incomplete selections, over-selections, or incorrect selections.
# - The output must be a valid Python list (only list, no explanation, no extra text).
# """

    # New Prompt
    PROMPT00 = f"""
    You are an assistant that performs **multi-label visual recognition** on an underwater RGB image.
    Before this RGB image, you have already been shown a **sonar reference image** of (approximately)
    the same area.

    Your task:
    - Carefully look at the **current RGB color image** and decide which of the following options
      actually appear in this RGB image.
    - You may use the **sonar reference** only when the RGB image is dark, noisy, or ambiguous,
      to help you confirm shapes and locations of objects.
    - If the sonar reference and the RGB image conflict, **always trust the RGB image.**

    Options:
    {OPTS}

    Detailed instructions:
    1. First, quickly read through all options so you know what kinds of objects to look for.
    2. Then, carefully scan the entire RGB image (center and all corners), checking:
       - large structures (e.g., ship, aircraft, pipeline, turbine),
       - smaller objects (e.g., robot, oil drums, electrical box),
       - partial/occluded objects that are still clearly recognizable.
    3. For **each** option in the list, internally decide whether it is:
       - **clearly present** in the RGB image,
       - **clearly absent**, or
       - **uncertain / ambiguous**.
       If you are uncertain because of low visibility or occlusion, you may re-check the sonar
       reference to help you decide.
    4. Only select options that are **clearly present** based on the combined evidence.
       - Do **not** guess.
       - Do **not** select an option if you are still uncertain after checking RGB and sonar.
    5. Ignore any objects that are not mentioned in the options list.

    Output format (very important):
    - You must output **only** a valid Python list of strings, with no explanation and no extra text.
    - The strings must match the option texts exactly.
    - For example:
      - If the ship and pipeline are present: ["sunken ship", "oil pipe"]
      - If none of the options appear: []

    Scoring rule reminder:
    - This is a multiple-choice question.
    - **All** correct options must be selected.
    - **No** incorrect options may be selected.
    - Incomplete selections, over-selections, or wrong selections are all scored as incorrect.

    Now, based on the **current RGB image** (optionally using the previous sonar image as an auxiliary cue),
    output only the final Python list of selected options.
    """
    # Argument Parser:

    p = argparse.ArgumentParser()
    # experiment parameters
    p.add_argument('--exp_name', type=str, default="Result_MVwSss_highLight_00")
    p.add_argument('--exp_idx', type=str, default='all')
    p.add_argument('--exp_json', type=str, default="/data/perception/highLight.json")
    p.add_argument('--images_dir', type=str, default="/data/perception/highLight")
    p.add_argument('--options', nargs='+', default=OPTS,)
    # model parameters
    p.add_argument('--model_template', type=str, default='OpenAI', choices=MODELS_TEMPLATE)
    p.add_argument('--model_name_or_path', type=str, required=True,)
    p.add_argument('--api_key', type=str, default=os.getenv("API_KEY"))
    p.add_argument('--base_url', type=str, default=os.getenv("BASE_URL"))
    # debug parameters
    p.add_argument('--details', type=bool, default=True, help='print more details in terminal')

    args = p.parse_args()

    # Initialization:

    saved_json_path = f"{args.exp_name}.json"
    saved_log_path = Path(f"{args.exp_name}.log")
    sys.stdout = open(saved_log_path, "w", encoding="utf-8")

    exp_json_path = args.exp_json
    with open(exp_json_path, "r", encoding="utf-8") as f:
        if args.exp_idx == 'all':
            all_exps = json.load(f)
        else:
            all_exps = {args.exp_idx: json.load(f)[args.exp_idx]}

    def get_img(images_dir: str, idx: int) -> str:
        """get image path by index"""
        return os.path.join(images_dir, f"G{idx}_FrontCamera.png")

    # Main Loop:

    final_score = 0
    all_log = {}
    for idx, exp in all_exps.items():
        exp_idx: int = int(idx)
        exp_gol: List[str] = exp['target']

        single_img = get_img(
            images_dir=args.images_dir,
            idx=idx,
        )

        if args.model_template == 'OpenAI':
            list_output, raw_output, full_messages = call_OpenAI(
                model_name_or_path=args.model_name_or_path,
                api_key=args.api_key,
                base_url=args.base_url,
                image_input=single_img,
                options=args.options,
                prompt=PROMPT00,
                prompt_sonar=PROMPT_SONAR,
            )
        elif args.model_template == 'Qwen':
            list_output, raw_output, full_messages = call_Qwen(
                device=DEVICE,
                model_name_or_path=args.model_name_or_path,
                image_input=single_img,
                options=args.options,
                prompt=PROMPT00,
                prompt_sonar=PROMPT_SONAR,
            )
        elif args.model_template == 'BMB':
            list_output, raw_output, full_messages = call_BMB(
                device=DEVICE,
                model_name_or_path=args.model_name_or_path,
                image_input=single_img,
                options=args.options,
                prompt=PROMPT00,
                prompt_sonar=PROMPT_SONAR,
            )
        elif args.model_template == 'Gemma':
            list_output, raw_output, full_messages = call_Gemma(
                device=DEVICE,
                model_name_or_path=args.model_name_or_path,
                image_input=single_img,
                options=args.options,
                prompt=PROMPT00,
                prompt_sonar=PROMPT_SONAR,
            )

        is_completely_match, matched_count = compare_eval(
            options=args.options,
            model_output=list_output,
            goal=exp_gol,
        )

        if is_completely_match:
            final_score += 1

        all_log[exp_idx] = {
            "img_path" : single_img,
            "messages": full_messages,
            "all_options": args.options,
            "target_option": exp_gol,
            "model_list_output": list_output,
            "is_completely_match": is_completely_match,
            "matched_count": matched_count,
        }
        with open(saved_json_path, "w", encoding="utf-8") as f:
            json.dump(all_log, f, indent=2, ensure_ascii=False)
        print(f">>> Progress saved to {saved_json_path} after completing question {idx}")

    # Count All:

    total = len(all_exps)
    acc = final_score / total

    all_log["summary"] = {
        "total_examples": total,
        "final_score": final_score,
        "accuracy": acc
    }
    with open(saved_json_path, "w", encoding="utf-8") as f:
        json.dump(all_log, f, indent=2, ensure_ascii=False)

    print(f">>> Final summary saved to {saved_json_path}")
    print(f"Total: {total}, Final score: {final_score}, Accuracy: {acc:.4f}")
