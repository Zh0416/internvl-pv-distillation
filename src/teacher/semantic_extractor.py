from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch

from .feature_extractor import load_pixel_values


def _normalise_probability(value: object) -> float:
    probability = float(value)
    if probability > 1.0:
        probability /= 100.0
    return max(0.0, min(1.0, probability))


def parse_semantic_response(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        exists_match = re.search(r"(?:pv_exists|存在|光伏).{0,20}(true|false|是|否)", text, re.I)
        probability_match = re.search(r"(?:pv_probability|光伏概率).{0,20}(\d+(?:\.\d+)?)", text, re.I)
        confidence_match = re.search(r"(?:confidence|置信度).{0,20}(\d+(?:\.\d+)?)", text, re.I)
        value = {"pv_exists": bool(exists_match and exists_match.group(1).lower() in {"true", "是"}),
                 "confidence": float(confidence_match.group(1)) if confidence_match else 0.0,
                 "pv_probability": float(probability_match.group(1)) if probability_match else None,
                 "reason": text[:500], "parse_warning": "fallback_parser"}
    raw_exists = value.get("pv_exists", False)
    if isinstance(raw_exists, str):
        pv_exists = raw_exists.strip().lower() in {"true", "1", "yes", "是"}
    else:
        pv_exists = bool(raw_exists)
    if value.get("pv_probability") is not None:
        pv_probability = _normalise_probability(value["pv_probability"])
        confidence = pv_probability if pv_exists else 1.0 - pv_probability
    else:
        confidence = _normalise_probability(value.get("confidence", 0.0))
        pv_probability = confidence if pv_exists else 1.0 - confidence
    result = {"pv_exists": pv_exists, "confidence": confidence,
              "pv_probability": pv_probability, "reason": str(value.get("reason", "")), "raw_output": raw}
    if pv_exists != (pv_probability >= 0.5):
        result["probability_warning"] = "pv_exists conflicts with pv_probability threshold 0.5"
    if value.get("parse_warning"):
        result["parse_warning"] = value["parse_warning"]
    return result


@torch.inference_mode()
def extract_semantic(bundle, image_path: str | Path, config: dict) -> tuple[dict, float]:
    started = time.perf_counter()
    model, tokenizer = bundle.model, bundle.tokenizer
    pixel_values, patch_count = load_pixel_values(image_path, model, config)
    generation_config = {"max_new_tokens": int(config["teacher"].get("max_new_tokens", 128)),
                         "do_sample": bool(config["teacher"].get("do_sample", False)),
                         "temperature": float(config["teacher"].get("temperature", 0.0)),
                         "top_p": float(config["teacher"].get("top_p", 1.0))}
    raw = model.chat(tokenizer, pixel_values, config["teacher"]["prompt"], generation_config,
                     num_patches_list=[patch_count], verbose=False)
    result = parse_semantic_response(raw)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["patch_count"] = patch_count
    del pixel_values
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, result["elapsed_seconds"]
