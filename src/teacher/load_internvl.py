from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class TeacherBundle:
    model: object
    tokenizer: object
    dtype: torch.dtype
    quantization: str


def _dtype(name: str) -> torch.dtype:
    if name.lower() == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def gpu_memory() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    return {"available": True, "device": torch.cuda.get_device_name(0),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)}


def load_teacher(config: dict, logger: logging.Logger) -> TeacherBundle:
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    dtype = _dtype(str(model_cfg.get("dtype", "float16")))
    revision = model_cfg.get("revision", "main")
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=trust_remote_code, use_fast=False)
    requested = str(model_cfg.get("quantization", "auto")).lower()
    attempts = ["8bit", "4bit", "none"] if requested == "auto" else [requested]
    last_error: Exception | None = None
    for quantization in attempts:
        kwargs = {"revision": revision, "trust_remote_code": trust_remote_code, "low_cpu_mem_usage": True,
                  "device_map": model_cfg.get("device_map", "auto"), "torch_dtype": dtype}
        if quantization == "8bit":
            kwargs["load_in_8bit"] = True
        elif quantization == "4bit":
            kwargs.update(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        try:
            logger.info("Loading %s with quantization=%s dtype=%s", model_name, quantization, dtype)
            model = AutoModel.from_pretrained(model_name, **kwargs).eval()
            logger.info("Teacher loaded; GPU memory: %s", gpu_memory())
            return TeacherBundle(model=model, tokenizer=tokenizer, dtype=dtype, quantization=quantization)
        except (RuntimeError, ValueError, ImportError, OSError) as exc:
            last_error = exc
            logger.exception("Teacher load failed with %s", quantization)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"无法加载InternVL Teacher；最后错误: {last_error}")
