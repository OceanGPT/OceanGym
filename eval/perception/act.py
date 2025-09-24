import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any

import base64
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoProcessor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_imgs(images_dir: str, directions: Set[str], idx: int) -> list[str]:
    """
    Get all images for a specific question index
    """
    prefix = f"G{idx}_"
    imgs = []

    for name in os.listdir(images_dir):
        if not name.startswith(prefix):
            continue
        direction = name[len(prefix):].split('.', 1)[0]
        if direction in directions:
            imgs.append(os.path.join(images_dir, name))
    return imgs


def encode_image_to_base64(image_path: str) -> str:
    """turn image to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def compare_answer(answer: str, goal: List[str], pattern: re.Pattern,) -> List[str]:
    return sorted({m.group(0).capitalize() for m in pattern.finditer(answer)})


def get_score(matching_list: List[str],) -> int:
    match_count = len(matching_list)
    if match_count == 0:
        return False, 0
    else:
        return True, match_count


def call_OpenAI(
        model_name_or_path: str,
        api_key: str,
        base_url: str,
        pattern: re.Pattern,
        image_input: List[str],
        text_input: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:

    messages = []
    for img in image_input:
        img_name = re.search(pattern, os.path.basename(img)).group(0).capitalize()
        img_b64 = encode_image_to_base64(img)

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{img_b64}",
                    },
                    {
                        "type": "input_text",
                        "text": f"This is {img_name} view.",
                    },
                ]
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": text_input
                }
            ]
        }
    )
    # print(f"messages: {messages}")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    resp = client.responses.create(
        model=model_name_or_path,
        input=messages,
        temperature=0,  # To ensure deterministic output
        top_p=1, # To ensure deterministic output
    )
    anwser = resp.output_text.strip()

    messages.append({"role": "assistant", "content": anwser})
    return anwser, messages


def call_Qwen(
        device: str,
        model_name_or_path: str,
        pattern: re.Pattern,
        image_input: List[str],
        text_input: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
    from transformers import Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path, torch_dtype="auto", device_map="auto"
    ).to(torch.device(device))
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        use_fast=False, # To ensure deterministic output
    )

    messages = []
    for img in image_input:
        img_name = re.search(pattern, os.path.basename(img)).group(0).capitalize()
        img_b64 = encode_image_to_base64(img)

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_url": f"data:image;base64,{img_b64}",
                    },
                    {
                        "type": "text",
                        "text": f"This is {img_name} view.",
                    },
                ]
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_input
                }
            ]
        }
    )
    # print(f"messages: {messages}")

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

    anwser = output_text[0].strip()
    messages.append({"role": "assistant", "content": anwser})
    return anwser, messages


def call_Gemma(
        device: str,
        model_name_or_path: str,
        pattern: re.Pattern,
        image_input: List[str],
        text_input: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
    from transformers import Gemma3ForConditionalGeneration

    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    messages = [
        {
            "role": "user",
            "content": [
                *[
                    {"type": "image", "image_url": img}
                    for img in image_input
                ],
                *[
                    {"type": "text", "text": f"This is {re.search(pattern, os.path.basename(img)).group(0).capitalize()} view."}
                    for img in image_input
                ],
                {"type": "text", "text": text_input}
            ]
        }
    ]

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
            max_new_tokens=100,
            do_sample=False,     # To ensure deterministic output
            temperature=0.0,     # To ensure deterministic output
            top_p=1.0, # To ensure deterministic output
            top_k=50, # To ensure deterministic output
        )
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

    anwser = decoded.strip()
    messages.append({"role": "assistant", "content": [{"type": "text", "text": anwser}]})
    return anwser, messages


if __name__ == "__main__":
    # Global Settings:

    set_seed(42)  # To ensure deterministic output

    DEVICE = "cuda:0"

    MODELS_TEMPLATE = ['OpenAI', 'Qwen', 'BMB', 'Gemma']

    DIRS = ["Front", "Back", "Left", "Right", "Up", "Down"]

    PROMPT_INSTRUCT = f"""You are an underwater robot navigation assistant responsible for analyzing images from various directions and making decisions based on problems.

Directions:
{DIRS}

Instructions:
- Carefully examine the image, even the corners.
- You can choose single or multiple options, if none of the directions appear, just return an empty Python list.
- The output must be a valid Python list (only list, no explanation, no extra text).
"""

    # Argument Parser:

    p = argparse.ArgumentParser()

    # experiment parameters
    p.add_argument('--exp_name', type=str, default='Result_Act_00',)
    p.add_argument('--exp_idx', type=str, default='all',)
    p.add_argument('--exp_json', type=str, default="/data/perception/highLight.json")
    p.add_argument('--images_dir', type=str, default='/data/perception/highLight')
    p.add_argument('--directions', nargs='+', default=DIRS,)
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
    if args.details: print(f"all_exps: {all_exps}")

    directions = set(args.directions)

    pattern = re.compile('|'.join(map(re.escape, directions)), re.IGNORECASE)

    # Main Loop:

    final_score = 0
    final_total_macth = 0
    all_dialogues = {}

    for idx, exp in all_exps.items():
        idx: int = int(idx)
        exp_Q: str = exp['prompt']
        exp_A: List[str] = exp['goal']
        print(f"=== @ This is No.{idx}, the question is: {exp_Q}, the answer is: {exp_A} ===")

        PROMPT_FULL = PROMPT_INSTRUCT + f"\nQuestion: {exp_Q}\nAnswer (in a Python list):"

        imgs = get_imgs(
            images_dir=args.images_dir,
            directions=directions,
            idx=idx,
        )
        if args.details: print(f"$ The images are: {imgs}")

        if args.model_template == 'OpenAI':
            anwser, full_traj = call_OpenAI(
                model_name_or_path=args.model_name_or_path,
                api_key=args.api_key,
                base_url=args.base_url,
                pattern=pattern,
                image_input=imgs,
                text_input=PROMPT_FULL,
            )
        elif args.model_template == 'Qwen':
            anwser, full_traj = call_Qwen(
                device=DEVICE,
                model_name_or_path=args.model_name_or_path,
                pattern=pattern,
                image_input=imgs,
                text_input=PROMPT_FULL,
            )
        elif args.model_template == 'Gemma':
            anwser, full_traj = call_Gemma(
                device=DEVICE,
                model_name_or_path=args.model_name_or_path,
                pattern=pattern,
                image_input=imgs,
                text_input=PROMPT_FULL,
            )
        if args.details: print(f"$ The model answer is: {anwser}")

        matching_list = compare_answer(
            answer=anwser,
            goal=exp_A,
            pattern=pattern,
        )
        if args.details: print(f"$ The matching result is: {matching_list}")

        match_flag, single_score = get_score(
            matching_list=matching_list,
        )

        print(f"=== @ The answer is: {anwser}\n~~~ Match Flag: {match_flag}, the single score is: {single_score} ===")

        final_score += single_score
        if match_flag:
            final_total_macth += 1

        all_dialogues[idx] = {
            "match_flag": match_flag,
            "question_images": imgs,
            "question_text": exp_Q,
            "answer_goal": exp_A,
            "answer_real_full": anwser,
            "score": single_score,
            "messages": full_traj,
        }

        with open(saved_json_path, "w", encoding="utf-8") as f:
            json.dump(all_dialogues, f, indent=2, ensure_ascii=False)
        print(f">>> Progress saved to {saved_json_path} after completing question {idx}")

    print(f"===== The final score is: {final_score} =====")

    total = len(all_exps)
    acc = final_total_macth / total

    all_dialogues["summary"] = {
        "total_examples": total,
        "final_total_match": final_total_macth,
        "final_score(not)": final_score,
        "accuracy": acc
    }

    with open(saved_json_path, "w", encoding="utf-8") as f:
        json.dump(all_dialogues, f, indent=2, ensure_ascii=False)

    print(f">>> Final summary saved to {saved_json_path}")
    print(f"Total: {total}, Final total match: {final_total_macth}, Final score (not): {final_score}, Accuracy: {acc:.4f}")
