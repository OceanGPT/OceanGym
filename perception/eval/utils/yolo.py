import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import os


def load_yolo_results(csv_path: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    读取 YOLO 的 detections_summary.csv，返回:
        { basename(image_path) : [(class_name, max_conf), ...] }，
    其中同一张图、同一类只保留最大置信度。
    """
    per_image_class_conf: Dict[Tuple[str, str], float] = {}

    if not os.path.exists(csv_path):
        print(f"[WARN] YOLO csv not found: {csv_path}, will not use tool predictions.")
        return {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row["image_path"]
            cls = row["class_name"]
            conf = float(row["confidence"])

            fname = os.path.basename(img_path)  # 例如 G10_sonar.png
            key = (fname, cls)

            # 同一张图同一类别，保留最大置信度
            if key not in per_image_class_conf or conf > per_image_class_conf[key]:
                per_image_class_conf[key] = conf

    # 转成： fname -> [(class, conf), ...] 并按置信度排序
    results: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (fname, cls), conf in per_image_class_conf.items():
        results[fname].append((cls, conf))

    for fname in results:
        results[fname].sort(key=lambda x: -x[1])  # 按置信度降序

    print(f"[INFO] Loaded YOLO results for {len(results)} images from {csv_path}.")
    return results

def build_sonar_prompt(
    sonar_img_path: str,
    base_prompt_sonar: str,
    yolo_results: Dict[str, List[Tuple[str, float]]],
) -> str:
    """
    Build the sonar prompt for the current image, optionally appending
    YOLO tool predictions in English.
    """
    fname = os.path.basename(sonar_img_path)  # e.g. G10_sonar.png
    dets = yolo_results.get(fname, None)

    if not dets:
        # No detections for this sonar image: just return the base prompt.
        return base_prompt_sonar

    # Optionally: filter out invalid/irrelevant classes
    # dets = [(cls, conf) for cls, conf in dets if cls in set(OPTS)]

    # Format as: plane (confidence 0.982); ship (confidence 0.945)
    parts = [f"{cls} (confidence {conf:.3f})" for cls, conf in dets]
    det_str = "; ".join(parts)

    extra = (
        "\nAdditionally, an external detection tool (YOLO) predicts "
        f"the following categories and probabilities for this sonar image: {det_str}. "
        "Treat these predictions only as auxiliary hints. "
        # "If they conflict with what you see in the sonar or RGB images, "
        # "always trust the visual evidence from the images."
    )

    return base_prompt_sonar + extra
