#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache neuron activations per layer for a VLMEvalKit-supported model on a chosen dataset.

This script blends two ideas:
- Load models via VLMEvalKit (supported_VLM) in the same safe manner as inference.py
- Load a meta probing dataset from Hugging Face (similar to my_llava-* script), and cache activations via forward hooks

Key features:
- Pick a VLMEvalKit-supported local model (API models are not supported since we need torch hooks)
- Choose "--hf-dataset meta" to build a mixed probing dataset (MMBench, VCR, DocVQA, VQAv2, ScienceQA, ST-VQA)
- Or fallback to VLMEvalKit datasets via --data
- Register forward hooks on selected target modules (regex + optional class filters)
- Run generation to trigger forwards; aggregate input/output activations (sum over token dimension) and average
- Save a dictionary {module_name: {input: 1D tensor, output: 1D tensor}}

Example:
  python cache_activation_new.py \
    --gpus 0,1,2,3 \
    --model mPLUG-Owl2 \
    --hf-dataset meta \
    --hf-offline \
    --req-act input output \
    --module-regex "mlp\\.|self_attn\\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn" \
    --probe-batch-size 1\
    --vlm-device-map auto \
    --vlm-dtype float16

Notes:
- Only local torch models are supported (API models cannot be hooked).
- Hooks try kwargs['hidden_states'] first or args[0] as input tensor.
- Output is assumed to be a Tensor or first element of a tuple, pooled by flattening to [tokens, hidden] and summing.
- When using --hf-dataset meta, images are saved to a tmp folder as files and referenced by path in messages.

Tip: run scripts/pre_download_hf_meta.py beforehand to cache HF datasets locally and avoid long waits at first run.

New (added):
- Support caching activations for pure-text LLMs loaded via Hugging Face transformers (e.g., meta-llama/Llama-2-7b-hf).
- Use only the textual part of datasets (questions + choices/hints) and ignore images.
- Example for LLM:
        python cache_activation_new.py \
            --hf-llm-id /root/autodl-tmp/AdaMMS/downloaded_models/Llama-2-7b-hf \
            --gpus 0,1,2,3 \
            --hf-dataset meta \
            --hf-offline \
            --req-act input output \
            --module-regex "mlp\\.|self_attn\\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn" \
            --probe-batch-size 1 --llm-max-length 512 \
            --llm-dtype float16 \
            --llm-device-map auto

Multi-GPU (new):
- For transformers LLM, pass --llm-device-map auto (or other maps) to shard the model across GPUs using accelerate.
- When device_map is set, the code will not .to(device) the whole model and will keep batch tensors on CPU to let HF dispatch them automatically.

单卡：使用物理 2 号卡 python cache_activation_new.py --gpus 2 --hf-llm-id /path/to/Llama-2-7b-hf --hf-dataset meta --n-mmbench 50 --req-act input output --module-regex "mlp.|self_attn.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn" --probe-batch-size 1 --llm-max-length 512

多卡：使用物理 2、6 号卡，并让 transformers 自动分片 python cache_activation_new.py --gpus 2,6 --hf-llm-id /path/to/Llama-2-7b-hf --hf-dataset meta --n-mmbench 50 --req-act input output --module-regex "mlp.|self_attn.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn" --probe-batch-size 1 --llm-max-length 512 --llm-device-map auto

"""



from __future__ import annotations

import argparse
import functools
import gc
import os
import os.path as osp
import random
import re
import uuid
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

# VLMEvalKit imports
from vlmeval.config import supported_VLM
try:
    from vlmeval.dataset import build_dataset as vlmeval_build_dataset
except Exception:
    vlmeval_build_dataset = None

# HF datasets
try:
    from datasets import load_dataset, DownloadConfig
except Exception as e:
    load_dataset = None
    _DATASETS_IMPORT_ERR = e
else:
    _DATASETS_IMPORT_ERR = None

try:
    from PIL import Image
except Exception:
    Image = None

# New: optional transformers import for pure LLM path
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

# Optional accelerate/transformers helpers for sharding VLM underlying HF model across GPUs
_acc_dispatch_model = None
_acc_get_balanced_memory = None
_acc_infer_auto_device_map = None
try:
    # Prefer accelerate top-level import when available
    from accelerate import dispatch_model as _acc_dispatch_model  # type: ignore
except Exception:
    try:
        from accelerate.utils import dispatch_model as _acc_dispatch_model  # type: ignore
    except Exception:
        _acc_dispatch_model = None

# get_balanced_memory / infer_auto_device_map exist in transformers.utils in many versions
try:
    from transformers.utils import get_balanced_memory as _acc_get_balanced_memory  # type: ignore
    from transformers.utils import infer_auto_device_map as _acc_infer_auto_device_map  # type: ignore
except Exception:
    try:
        from accelerate.utils import (
            get_balanced_memory as _acc_get_balanced_memory,  # type: ignore
            infer_auto_device_map as _acc_infer_auto_device_map,  # type: ignore
        )
    except Exception:
        _acc_get_balanced_memory = None
        _acc_infer_auto_device_map = None
# -------------------------
# CLI
# -------------------------

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache layer activations for a VLM on a dataset (meta/HF or VLMEval)")
    # model/dataset
    # Note: either --model (VLMEvalKit) or --hf-llm-id (transformers) should be provided
    parser.add_argument("--model", required=False, type=str, help="Model name key in supported_VLM (vlmeval/config.py)")
    parser.add_argument("--hf-llm-id", type=str, default=None,
                        help="HuggingFace model id or local path for transformers AutoModelForCausalLM; e.g., meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data", required=False, type=str, default=None,
                        help="Dataset name supported by VLMEvalKit (ignored if --hf-dataset is set)")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="Special HF loader key; currently supports: 'meta'. If set, overrides --data")

    # meta dataset knobs
    parser.add_argument("--n-mmbench", type=int, default=40, help="Samples to draw from MMBench (en/test)")
    parser.add_argument("--n-vcr", type=int, default=0, help="Samples to draw from VCR (validation, Q->A)")
    parser.add_argument("--n-docvqa", type=int, default=10, help="Samples to draw from DocVQA (validation)")
    parser.add_argument("--n-vqa", type=int, default=50, help="Samples to draw from VQAv2 (validation)")
    parser.add_argument("--n-scienceqa", type=int, default=50, help="Samples to draw from ScienceQA (validation, has image)")
    parser.add_argument("--n-stvqa", type=int, default=50, help="Samples to draw from ST-VQA task1 (test)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap total samples (after composition)")
    # 新增：HF 下载与镜像控制
    parser.add_argument("--hf-no-streaming", action="store_true",
                        help="Disable datasets streaming mode; use regular download (more stable, larger download).")
    parser.add_argument("--hf-endpoint", type=str, default=None,
                        help="Override HF_ENDPOINT (e.g., https://huggingface.co or a mirror). Use 'disable' to unset.")
    parser.add_argument("--hf-offline", action="store_true",
                        help="Force offline mode for HF datasets (use local cache only); implies --hf-no-streaming.")
    # 新增：缓存目录与回退控制
    parser.add_argument("--hf-cache-dir", type=str, default=None,
                        help="Override HuggingFace cache dir for datasets (sets HF_HOME/HF_DATASETS_CACHE/HUGGINGFACE_HUB_CACHE and DownloadConfig.cache_dir).")
    parser.add_argument("--hf-disable-streaming-fallback", action="store_true",
                        help="Disable automatic fallback to streaming when non-streaming fails due to disk space.")

    # hook + selection
    parser.add_argument("--req-act", nargs="+", default=["output"], choices=["input", "output"],
                        help="Which activations to record: input/output (one or both)")
    parser.add_argument("--module-regex", type=str,
                        default=r"mlp\. |self_attn\.|attention\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn".replace(" ", ""),
                        help="Regex to select modules by name. Applied to named_modules() full path.")
    parser.add_argument("--include-types", nargs="*", default=["Linear"],
                        help="Optional nn.Module class name filters, e.g. Linear Conv2d LayerNorm; empty=all")
    parser.add_argument("--exclude-regex", type=str, default=r"lm_head|embed|embedding",
                        help="Regex to exclude modules by name")

    # misc
    parser.add_argument("--work-dir", type=str, default=".", help="Work dir for tmp files")
    parser.add_argument("--save", type=str, default=None, help="Output .pt file path; default under activations/")
    parser.add_argument("--verbose", action="store_true", help="Print progress and matched modules")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Pass use_vllm to certain models (e.g., Llama-4, Qwen2-VL series)")

    # GPU selection
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids to use, e.g. '0,2'. Will set CUDA_VISIBLE_DEVICES accordingly.")

    # New: transformers LLM runtime knobs
    parser.add_argument("--llm-device", type=str, default="cuda",
                        help="Device for transformers LLM (cuda/cpu). If using --gpus, the indices refer to the remapped list.")
    parser.add_argument("--llm-dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="torch dtype for transformers LLM load; auto=HF default")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code=True to transformers.from_pretrained")
    parser.add_argument("--probe-batch-size", type=int, default=1,
                        help="Batch size for forward pass when using transformers LLM (text-only)")
    parser.add_argument("--llm-max-length", type=int, default=512,
                        help="Max token length for tokenizer(truncation)")
    parser.add_argument("--llm-device-map", type=str, default=None,
                        help="Transformers device_map for multi-GPU sharding, e.g. 'auto', 'balanced', 'balanced_low_0'. Use None to disable.")
    # 与 VLM 一致：默认使用 generate 路径触发模块前向
    parser.add_argument("--llm-forward-mode", type=str, choices=["generate", "forward"], default="generate",
                        help="LLM 前向路径：'generate' 与 VLM 一致（推荐），或 'forward' 直接调用 model(**enc)。默认 generate。")
    parser.add_argument("--llm-new-tokens", type=int, default=1,
                        help="当使用 --llm-forward-mode generate 时，生成的新 token 数（建议 1）。")

    # VLM (VLMEvalKit) optional multi-GPU sharding via accelerate
    parser.add_argument("--vlm-device-map", type=str, default="auto",
                        help="Optional device map for underlying HF model inside VLM wrappers. Use 'auto' to infer and shard across visible GPUs. If None, keep default single-device placement by wrapper.")
    parser.add_argument("--vlm-dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Dtype hint used when inferring device map (only takes effect if --vlm-device-map is set).")
    parser.add_argument("--vlm-no-split-classes", nargs="*", default=[
        "LlamaDecoderLayer", "MistralDecoderLayer", "QWenDecoderLayer", "Qwen2DecoderLayer", "Qwen2VLDecoderLayer"
    ], help="Class names that should not be split when inferring device map with accelerate.")

    # FAM: FAI computation knobs
    parser.add_argument("--fai-compute", action="store_true",
                        help="Compute Flatness-Activation Importance (FAI) per neuron for each hooked module.")
    parser.add_argument("--fai-max-samples-per-module", type=int, default=4096,
                        help="Max per-sample rows to keep for MI/Hessian computation per module (to bound memory).")
    parser.add_argument("--fai-mi-mode", type=str, choices=["mi", "pearson", "spearman"], default="mi",
                        help="Mutual dependence measure for FAI: sklearn mutual_info_regression (mi), Pearson |r|, or Spearman |rho|.")
    parser.add_argument("--fai-eps", type=float, default=1e-6,
                        help="Stability epsilon used in denominator of FAI formula.")

    # LAPE Stage-1: path-sampling based length-agnostic encoding
    parser.add_argument("--lape-enable", action="store_true",
                        help="Enable LAPE Stage-1: per-sample multi-path sampling to compute phi=[a~, ḡ, L̄] per module.")
    parser.add_argument("--lape-samples", type=int, default=8,
                        help="Number of generation paths per input sample for LAPE (N_s).")
    parser.add_argument("--lape-gamma", type=float, default=0.99,
                        help="Geometric discount factor for path length in LAPE (gamma).")
    parser.add_argument("--lape-top-p", type=float, default=0.9,
                        help="Top-p nucleus sampling parameter when sampling generation paths.")
    parser.add_argument("--lape-temperature", type=float, default=0.7,
                        help="Sampling temperature used for LAPE path sampling (applies to both LLaVA/mPLUG).")
    parser.add_argument("--lape-min-new", type=int, default=1,
                        help="Minimum number of new tokens to sample per path.")
    parser.add_argument("--lape-max-new", type=int, default=8,
                        help="Maximum number of new tokens to sample per path.")
    parser.add_argument("--lape-yref", type=str, choices=["zero", "running_avg"], default="running_avg",
                        help="Reference target Y_ref used in gradient proxy: zero or running_avg of module outputs.")
    parser.add_argument("--lape-deterministic", action="store_true",
                        help="Deterministic LAPE: 禁止随机长度与随机采样；使用 temperature=0, top_p=1, 固定 new_tokens = lape-max-new。")

    # LAPE per-sample MI support
    parser.add_argument("--lape-mi-store", action="store_true",
                        help="Store per-input (phi_a, phi_g, phi_L) samples for each module to enable per-neuron MI estimation in Stage-2 (memory intensive).")
    parser.add_argument("--lape-mi-max-samples", type=int, default=512,
                        help="Max number of per-input phi samples to retain per module (uniform cap). Use -1 for unlimited (may consume large memory).")

    return parser.parse_args()


# -------------------------
# Meta dataset builder (HF)
# -------------------------

def _ensure_hf_import():
    if load_dataset is None:
        raise RuntimeError(
            f"datasets is not available for --hf-dataset; please install `datasets`. Import error: {_DATASETS_IMPORT_ERR}"
        )


def _dump_image_to_file(img: Any, root: str) -> str:
    os.makedirs(root, exist_ok=True)
    # Convert to RGB if it's a PIL Image with alpha
    if Image is not None and isinstance(img, Image.Image):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            try:
                img = img.convert('RGB')
            except Exception:
                pass
    # save
    fname = f"{uuid.uuid4().hex}.jpg"
    path = osp.join(root, fname)
    try:
        if Image is not None and isinstance(img, Image.Image):
            img.save(path, format='JPEG', quality=95)
        else:
            # datasets Image feature returns PIL Image; if not, try array-like
            from PIL import Image as _PILImage
            _PILImage.fromarray(img).save(path, format='JPEG', quality=95)
    except Exception as e:
        raise RuntimeError(f"Failed to save image to {path}: {e}")
    return path


def build_meta_probe_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Compose a meta probing dataset by sampling from several HF datasets.

    Returns list of dict with keys: {image (PIL), question (str), optional answer/answers}.
    """
    _ensure_hf_import()
    # 若指定端点，优先设置（可覆盖全局环境的镜像设置）
    if args.hf_endpoint is not None:
        if args.hf_endpoint.strip().lower() == "disable":
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = args.hf_endpoint.strip()

    # Offline/streaming config
    offline = bool(getattr(args, "hf_offline", False))
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    # 新增：缓存目录设置
    if args.hf_cache_dir:
        # Expand ~ and make absolute to avoid writing to system default caches
        _cache_dir = osp.abspath(osp.expanduser(args.hf_cache_dir))
        os.makedirs(_cache_dir, exist_ok=True)
        # Point all relevant caches and temp dir to the specified location
        os.environ["HF_HOME"] = _cache_dir
        os.environ["HF_DATASETS_CACHE"] = _cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = _cache_dir
        os.environ["TRANSFORMERS_CACHE"] = _cache_dir
        # Redirect temp files (parquet/arrow/temp extractions) away from system /tmp
        os.environ.setdefault("TMPDIR", _cache_dir)
    else:
        _cache_dir = None

    streaming = False if offline else (not getattr(args, "hf_no_streaming", False))
    dl_cfg = DownloadConfig(
        max_retries=0 if offline else 10,
        resume_download=not offline,
        use_etag=not offline,
        local_files_only=offline,
        cache_dir=_cache_dir,
    )

    def _set_ep(ep: Optional[str]):
        if ep is None:
            return
        if ep.strip().lower() == "disable":
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = ep.strip()

    # 新增：判定是否是“磁盘空间不足”错误
    def _is_disk_oom(exc: Exception) -> bool:
        s = str(exc).lower()
        return ("not enough disk space" in s) or ("no space left on device" in s) or ("errno 28" in s)

    def _streaming_order() -> List[bool]:
        # 请求顺序：若用户要求 streaming，则优先 True；否则先 False，再视情况回退 True
        order = []
        if streaming:
            order.extend([True, False])
        else:
            order.append(False)
            if (not offline) and (not args.hf_disable_streaming_fallback):
                order.append(True)
        return order

    def _load_ds(path, name=None, split="validation"):
        current_ep = os.environ.get("HF_ENDPOINT")
        trial_eps = [current_ep, "https://huggingface.co"]
        seen = set(); ordered_eps = []
        for ep in trial_eps:
            key = ep or "__none__"
            if key in seen: continue
            seen.add(key); ordered_eps.append(ep)

        last_exc = None
        for ep in ordered_eps:
            _set_ep(ep)
            ep_str = ep or os.environ.get("HF_ENDPOINT") or "<default>"
            for sflag in _streaming_order():
                try:
                    print(f"[Meta] try load {path} ({name or ''}) split={split} streaming={sflag} ep={ep_str}")
                    if name is None:
                        return load_dataset(path, split=split, streaming=sflag, download_config=None if sflag else dl_cfg)
                    else:
                        return load_dataset(path, name, split=split, streaming=sflag, download_config=None if sflag else dl_cfg)
                except Exception as e:
                    last_exc = e
                    print(f"[Meta][warn] load failed at ep={ep_str} streaming={sflag}: {type(e).__name__}: {e}")
                    # 非流式因磁盘不足 -> 直接尝试流式一次（即便用户传了 --hf-no-streaming），除非用户禁用了回退
                    if (not sflag) and _is_disk_oom(e) and (not offline) and (not args.hf_disable_streaming_fallback):
                        try:
                            print(f"[Meta] disk full; fallback to streaming=True ep={ep_str}")
                            if name is None:
                                return load_dataset(path, split=split, streaming=True, download_config=None)
                            else:
                                return load_dataset(path, name, split=split, streaming=True, download_config=None)
                        except Exception as e2:
                            last_exc = e2
                            print(f"[Meta][warn] fallback streaming load failed: {type(e2).__name__}: {e2}")
                    continue
        raise RuntimeError(f"[Meta] Failed to load {path} ({name or ''}) split={split} after retries: {last_exc}")

    def _yield_first_n(path: str, name: Optional[str], split: str, n: int):
        """Robustly yield first n items with retries across endpoint and streaming modes.

        This handles failures that occur during iteration (common in streaming or when non-streaming hits disk limits)
        by switching modes and/or endpoints.
        """
        if n <= 0:
            return
        current_ep = os.environ.get("HF_ENDPOINT")
        trial_eps = [current_ep, "https://huggingface.co"]
        seen = set(); ordered_eps = []
        for ep in trial_eps:
            key = ep or "__none__"
            if key in seen: continue
            seen.add(key); ordered_eps.append(ep)

        for ep in ordered_eps:
            _set_ep(ep)
            ep_str = ep or os.environ.get("HF_ENDPOINT") or "<default>"
            for sflag in _streaming_order():
                try:
                    print(f"[Meta] try iterate {path} ({name or ''}) split={split} streaming={sflag} ep={ep_str}")
                    if name is None:
                        ds = load_dataset(path, split=split, streaming=sflag, download_config=None if sflag else dl_cfg)
                    else:
                        ds = load_dataset(path, name, split=split, streaming=sflag, download_config=None if sflag else dl_cfg)
                    cnt = 0
                    if sflag:
                        for item in ds.shuffle(seed=42).take(n):
                            yield item
                            cnt += 1
                        if cnt >= n:
                            return
                    else:
                        dss = ds.shuffle(seed=42)
                        n_eff = min(n, len(dss))
                        if n_eff <= 0:
                            return
                        for item in dss.select(range(n_eff)):
                            yield item
                        return
                except Exception as e:
                    print(f"[Meta][warn] iterate failed at ep={ep_str} streaming={sflag}: {type(e).__name__}: {e}")
                    # 非流式迭代时磁盘不足 -> 尝试流式回退
                    if (not sflag) and _is_disk_oom(e) and (not offline) and (not args.hf_disable_streaming_fallback):
                        try:
                            print(f"[Meta] disk full on iterate; retry with streaming=True ep={ep_str}")
                            if name is None:
                                ds = load_dataset(path, split=split, streaming=True, download_config=None)
                            else:
                                ds = load_dataset(path, name, split=split, streaming=True, download_config=None)
                            for item in ds.shuffle(seed=42).take(n):
                                yield item
                            return
                        except Exception as e2:
                            print(f"[Meta][warn] streaming iterate fallback failed: {type(e2).__name__}: {e2}")
                            continue
                    continue
        raise RuntimeError(f"[Meta] Failed to iterate dataset {path} ({name or ''}) split={split} for first {n} samples after retries")

    meta_probe_samples: List[Dict[str, Any]] = []

    # 1) MMBench EN (test)
    if getattr(args, 'n_mmbench', 0) > 0:
        print(f"[Meta] Loading {args.n_mmbench} from MMBench (en/test)...")
        for item in _yield_first_n("lmms-lab/MMBench", "en", "test", args.n_mmbench):
            q = item['question']
            options = []
            for key in ['A', 'B', 'C', 'D', 'E', 'F']:
                if key in item and item[key] is not None:
                    options.append(f"{key}. {item[key]}")
            options_str = "\n".join(options)
            if item.get('hint'):
                full_q = f"{item['hint']}\n{q}\n{options_str}" if options_str else f"{item['hint']}\n{q}"
            else:
                full_q = f"{q}\n{options_str}" if options_str else q
            meta_probe_samples.append({
                "image": item["image"],
                "question": full_q,
                "answer": item.get("answer", None)
            })

    # 2) VCR (validation, Q->A)
    if getattr(args, 'n_vcr', 0) > 0:
        print(f"[Meta] Loading {args.n_vcr} from VCR (validation, Q->A)...")
        for item in _yield_first_n("pingzhili/vcr-qa", None, "validation", args.n_vcr):
            q = item['question']
            choices = item.get('answer_choices', [])
            choices_str = "\n".join([f"- {c}" for c in choices])
            full_q = f"{q}\n\nChoices:\n{choices_str}" if choices_str else q
            label = item.get('answer_label', None)
            correct_text = choices[label] if (isinstance(label, int) and 0 <= label < len(choices)) else None
            meta_probe_samples.append({
                "image": item["image"],
                "question": full_q,
                "answer": correct_text
            })

    # 3) DocVQA (validation)
    if getattr(args, 'n_docvqa', 0) > 0:
        print(f"[Meta] Loading {args.n_docvqa} from DocVQA (validation)...")
        for item in _yield_first_n("lmms-lab/DocVQA", "DocVQA", "validation", args.n_docvqa):
            meta_probe_samples.append({
                "image": item["image"],
                "question": item["question"],
                "answers": item.get("answers", None)
            })

    # 4) VQAv2 (validation)
    if getattr(args, 'n_vqa', 0) > 0:
        print(f"[Meta] Loading {args.n_vqa} from VQAv2 (validation)...")
        for item in _yield_first_n("lmms-lab/VQAv2", None, "validation", args.n_vqa):
            meta_probe_samples.append({
                "image": item["image"],
                "question": item["question"],
            })

    # 5) ScienceQA (validation, has image) 仅非流式
    if getattr(args, 'n_scienceqa', 0) > 0:
        print(f"[Meta] Loading {args.n_scienceqa} from ScienceQA (validation, has image)...")
        ds = _load_ds("derek-thomas/ScienceQA", None, "validation")
        if streaming:
            # streaming 模式下 filter 不便，直接顺序筛选
            cnt = 0
            for item in ds:
                if item.get('image') is None:
                    continue
                hint = item.get('hint', None)
                q = item.get('question', '')
                full_q = f"{hint} {q}" if hint else q
                meta_probe_samples.append({"image": item["image"], "question": full_q})
                cnt += 1
                if cnt >= args.n_scienceqa:
                    break
        else:
            ds_img = ds.filter(lambda x: x.get('image') is not None)
            dss = ds_img.shuffle(seed=42)
            n_eff = min(args.n_scienceqa, len(dss))
            for item in dss.select(range(n_eff)):
                hint = item.get('hint', None)
                q = item.get('question', '')
                full_q = f"{hint} {q}" if hint else q
                meta_probe_samples.append({"image": item["image"], "question": full_q})
        del ds

    # 6) ST-VQA task1 (test)
    if getattr(args, 'n_stvqa', 0) > 0:
        print(f"[Meta] Loading {args.n_stvqa} from ST-VQA task1 (test)...")
        for item in _yield_first_n("danjacobellis/stvqa_task1", None, "test", args.n_stvqa):
            meta_probe_samples.append({"image": item["image"], "question": item["question"]})

    random.shuffle(meta_probe_samples)
    if args.max_samples is not None:
        meta_probe_samples = meta_probe_samples[: args.max_samples]
    print(f"[Meta] Built meta probing dataset, total samples: {len(meta_probe_samples)}")
    return meta_probe_samples


# -------------------------
# Utilities for text-only prompts (LLM path)
# -------------------------

def _get_texts_from_meta_samples(samples: List[Dict[str, Any]]) -> List[str]:
    """Extract text prompts from meta samples (ignore images).

    Each sample has 'question' already including choices/hints if applicable.
    """
    texts = []
    for s in samples:
        q = s.get("question", "")
        if not isinstance(q, str):
            q = str(q)
        texts.append(q)
    return texts

def _extract_texts_from_vlmeval_dataset(dataset, max_n: Optional[int]) -> List[str]:
    """Build text prompts from a VLMEval dataset by concatenating text segments in messages.

    This is a best-effort extraction: we collect all dicts with type='text' and join them.
    """
    data = dataset.data
    max_n = len(data) if max_n is None else min(len(data), max_n)
    texts: List[str] = []
    for i in range(max_n):
        struct = dataset.build_prompt(data.iloc[i])  # list of {type: 'image'|'text', value:...}
        parts = []
        for seg in struct:
            if isinstance(seg, dict) and seg.get("type") == "text":
                parts.append(str(seg.get("value", "")))
        texts.append("\n".join([p for p in parts if p]))
    return texts


# -------------------------
# Torch model inspection & selection
# -------------------------

def get_underlying_torch_model(vlm_obj) -> Optional[nn.Module]:
    """Try to retrieve the underlying torch.nn.Module from a VLMEvalKit model wrapper.

    Many wrappers use attribute `model` to hold the HF/torch model. If not present but the
    wrapper itself is an nn.Module, return the wrapper. Otherwise return None.
    """
    if hasattr(vlm_obj, "model") and isinstance(getattr(vlm_obj, "model"), nn.Module):
        return getattr(vlm_obj, "model")
    if isinstance(vlm_obj, nn.Module):
        return vlm_obj
    return None


def _class_name(m: nn.Module) -> str:
    return m.__class__.__name__


def get_target_module_map(
    model: nn.Module,
    module_regex: str,
    include_types: List[str],
    exclude_regex: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, nn.Module]:
    pat = re.compile(module_regex)
    ex_pat = re.compile(exclude_regex) if exclude_regex else None
    allow_set = set(include_types or [])

    matched: Dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if name == "":
            continue
        if not pat.search(name):
            continue
        if ex_pat and ex_pat.search(name):
            continue
        if allow_set and _class_name(mod) not in allow_set:
            continue
        matched[name] = mod

    if verbose:
        print(f"[Hook] Matched {len(matched)} modules:")
        for n, m in matched.items():
            print(f" - {n} ({_class_name(m)})")
    return matched


# -------------------------
# Device helpers (LLM)
# -------------------------

def _pick_llm_input_device(model: nn.Module) -> torch.device:
    """Choose a device to place input tensors for generate/forward.

    - If the model is sharded (hf_device_map exists), pick the lowest-index CUDA device in the map.
    - Else, use the device of the first parameter (covers single-GPU or CPU).
    - Fallback to CPU if anything goes wrong.
    """
    try:
        # Prefer device from device map if present
        if hasattr(model, 'hf_device_map') and isinstance(getattr(model, 'hf_device_map'), dict):
            dev_indices = []
            for v in model.hf_device_map.values():
                if isinstance(v, str) and v.startswith('cuda'):
                    try:
                        idx = int(v.split(':')[1]) if ':' in v else 0
                        dev_indices.append(idx)
                    except Exception:
                        continue
            if dev_indices:
                return torch.device(f'cuda:{min(dev_indices)}')
        # Fall back to first param device
        try:
            p = next(model.parameters())
            if hasattr(p, 'device'):
                return p.device
        except StopIteration:
            pass
    except Exception:
        pass
    return torch.device('cpu')


# -------------------------
# Hook function (input/output sum & average)
# -------------------------

def get_hook_with_kwargs(name: str, req_act: Iterable[str], activation_stats: dict, *, keep_batches: bool = False):
    def hook_fn(module, args, kwargs, output):
        # Lazy init extended sample buffers (for FAI) only when requested
        if keep_batches:
            if "input_batches" not in activation_stats[name]:
                activation_stats[name]["input_batches"] = []
            if "output_batches" not in activation_stats[name]:
                activation_stats[name]["output_batches"] = []

        # helper: pool to [batch, hidden]
        def _to_bh(t: torch.Tensor) -> Optional[torch.Tensor]:
            try:
                if t is None:
                    return None
                if not isinstance(t, torch.Tensor):
                    return None
                tt = t.detach().to("cpu", non_blocking=True).float()
                if tt.dim() == 3:
                    # assume [B, T, H] and pool tokens -> sum over T
                    if tt.shape[-1] <= 0:
                        return None
                    return tt.sum(dim=1)
                elif tt.dim() == 2:
                    # treat as [N, H]
                    return tt
                elif tt.dim() == 1:
                    return tt.unsqueeze(0)
                else:
                    # fallback: flatten all leading dims as batch
                    h = tt.shape[-1]
                    return tt.reshape(-1, h)
            except Exception:
                return None

        # Output
        if "output" in req_act:
            out_tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(out_tensor, torch.Tensor):
                t_float = out_tensor.detach().cpu().float()
                try:
                    t_reshaped = t_float.reshape(-1, t_float.shape[-1])
                except Exception:
                    return
                current_sum = torch.sum(t_reshaped, dim=0)
                if activation_stats[name]["output_sum"] is None:
                    activation_stats[name]["output_sum"] = current_sum
                else:
                    activation_stats[name]["output_sum"] += current_sum
                activation_stats[name]["output_tokens"] += t_reshaped.shape[0]
                # store per-sample pooled batch rows for FAI
                if keep_batches:
                    bh = _to_bh(out_tensor)
                    if bh is not None:
                        activation_stats[name]["output_batches"].append(bh)
        # Input
        if "input" in req_act:
            in_tensor = kwargs.get("hidden_states", args[0] if (args and isinstance(args[0], torch.Tensor)) else None)
            if isinstance(in_tensor, torch.Tensor):
                t_float = in_tensor.detach().cpu().float()
                try:
                    t_reshaped = t_float.reshape(-1, t_float.shape[-1])
                except Exception:
                    return
                current_sum = torch.sum(t_reshaped, dim=0)
                if activation_stats[name]["input_sum"] is None:
                    activation_stats[name]["input_sum"] = current_sum
                else:
                    activation_stats[name]["input_sum"] += current_sum
                activation_stats[name]["input_tokens"] += t_reshaped.shape[0]
                # store per-sample pooled batch rows for FAI
                if keep_batches:
                    bh = _to_bh(in_tensor)
                    if bh is not None:
                        activation_stats[name]["input_batches"].append(bh)
    return hook_fn


# -------------------------
# LAPE Stage-1 helpers (LLM only)
# -------------------------

class _LAPETracker:
    """Collect per-path pooled activations for selected modules and compute phi per input.

    Strategy:
    - During a single generate() call (one path), hooks append pooled per-forward outputs to buffers.
    - After the call, we aggregate buffers per module into vectors (sum over batch rows and forwards).
    - We then combine across multiple paths using weights w_k = softmax(log p(S_k|x) + (len_k-1) * log(gamma)).

    We treat module output pooled vector as both activation a and Y for gradient proxy.
    """
    def __init__(self, module_names: List[str], req_act: Iterable[str], gamma: float, yref_mode: str = "running_avg",
                 store_samples: bool = False, mi_max_samples: int = 512):
        self.module_names = list(module_names)
        self.req_act = set(req_act)
        self.gamma = float(gamma)
        self.yref_mode = yref_mode
        # per-path temporary buffers
        self._path_buf: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        # running reference Y per module
        self.y_ref: Dict[str, torch.Tensor] = {}
        self.y_ref_count: Dict[str, int] = defaultdict(int)
        # dataset accumulators (sum over inputs of phi(x))
        self.sum_phi_a: Dict[str, torch.Tensor] = {}
        self.sum_phi_g: Dict[str, torch.Tensor] = {}
        self.sum_phi_L: Dict[str, torch.Tensor] = {}
        self.n_inputs: int = 0
        # per-sample storage for MI (optional)
        self.store_samples = bool(store_samples)
        self.mi_max_samples = int(mi_max_samples)
        self.samples_phi_a: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.samples_phi_g: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.samples_phi_L: Dict[str, List[torch.Tensor]] = defaultdict(list)
        # count attempts for potential future reservoir logic (simple cap now)
        self._samples_attempts: Dict[str, int] = defaultdict(int)

    def start_path(self):
        # reset buffers for a new sampled path
        self._path_buf = {m: {"out": [], "inp": []} for m in self.module_names}

    def hook_record(self, name: str, in_bh: Optional[torch.Tensor], out_bh: Optional[torch.Tensor]):
        if name not in self._path_buf:
            return
        if out_bh is not None:
            self._path_buf[name]["out"].append(out_bh.detach().cpu().float())
        if in_bh is not None and ("input" in self.req_act):
            self._path_buf[name]["inp"].append(in_bh.detach().cpu().float())

    @staticmethod
    def _sum_rows(tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not tensors:
            return None
        try:
            cat = torch.cat(tensors, dim=0)  # [N_total, H]
            return cat.sum(dim=0)  # [H]
        except Exception:
            # Fallback: sequential sum
            acc = None
            for x in tensors:
                v = x
                acc = v.sum(dim=0) if acc is None else acc + v.sum(dim=0)
            return acc

    def _finalize_single_path_vectors(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Aggregate per-path buffers into vectors per module.

        Returns: {module: {"a": Tensor[H], "y": Tensor[H]}}
        We use output pooled vector as both activation a and Y for gradient proxy.
        If outputs are missing, returns empty dict for that module.
        """
        result: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in self.module_names:
            out_vec = self._sum_rows(self._path_buf[name]["out"]) if name in self._path_buf else None
            if out_vec is None:
                continue
            # For input activations, you could also use inp_vec for alternative a, but default to output
            a_vec = out_vec
            y_vec = out_vec
            result[name] = {"a": a_vec, "y": y_vec}
        return result

    def accumulate_one_input(self, path_records: List[Tuple[float, int, Dict[str, Dict[str, torch.Tensor]]]]):
        """Combine multiple paths for one input into phi(x) and add to dataset sums.

        path_records: list of (log_p, len_gen, per_module_vectors)
        Weight per path is softmax over log(w_k) where w_k = p(S_k|x) * gamma^{len_k-1}.
        """
        if not path_records:
            return
        # compute normalized weights for stability
        logws = []
        for (log_p, Lk, _vecs) in path_records:
            logw = float(log_p) + float(max(Lk - 1, 0)) * float(np.log(max(self.gamma, 1e-8)))
            logws.append(logw)
        logws_t = torch.tensor(logws, dtype=torch.float32)
        ws = torch.softmax(logws_t, dim=0).tolist()

        # union of modules that have vectors in any path
        mod_set = set()
        for _, _, vecs in path_records:
            mod_set.update(vecs.keys())

        for name in mod_set:
            phi_a = None
            phi_g = None
            phi_L = None
            # prepare y_ref
            yref = self.y_ref.get(name, None)

            for idx, (_, _Lk, vecs) in enumerate(path_records):
                if name not in vecs:
                    continue
                w = ws[idx]
                a_vec = vecs[name]["a"]  # [H]
                y_vec = vecs[name]["y"]  # [H]
                # gradient proxy per-neuron: a_i * (y_i - y_ref_i)
                if yref is None or self.yref_mode == "zero":
                    g_vec = a_vec * y_vec
                else:
                    # align device/dtype
                    yy = y_vec
                    if yref.device != yy.device:
                        yy = yy.to(yref.device)
                    g_vec = a_vec * (yy - yref)

                # L̄_k is unknown per module; broadcast scalar later (set to zeros here; will be overwritten outside)
                # Accumulate weighted sums
                phi_a = a_vec * w if phi_a is None else phi_a + a_vec * w
                phi_g = g_vec * w if phi_g is None else phi_g + g_vec * w

            # we defer L aggregation until caller supplies weighted scalar; leave placeholder zeros
            if phi_a is not None:
                H = phi_a.shape[0]
                phi_L = torch.zeros(H, dtype=phi_a.dtype)
                # dataset accumulators
                if name not in self.sum_phi_a:
                    self.sum_phi_a[name] = phi_a.clone()
                    self.sum_phi_g[name] = phi_g.clone() if phi_g is not None else torch.zeros_like(phi_a)
                    self.sum_phi_L[name] = phi_L.clone()
                else:
                    self.sum_phi_a[name] += phi_a
                    self.sum_phi_g[name] += phi_g if phi_g is not None else 0.0
                    self.sum_phi_L[name] += phi_L
                # optional per-sample storage (phi_a / phi_g). L will be added later in accumulate_L_scalar.
                if self.store_samples:
                    self._maybe_store_sample(name, 'a', phi_a)
                    if phi_g is not None:
                        self._maybe_store_sample(name, 'g', phi_g)

        # count one input
        self.n_inputs += 1

    def accumulate_L_scalar(self, name: str, L_scalar_weighted: float, H: int):
        # Add weighted loss scalar to phi_L accumulator (broadcast along neurons)
        vec = torch.full((H,), float(L_scalar_weighted), dtype=torch.float32)
        if name not in self.sum_phi_L:
            self.sum_phi_L[name] = vec
        else:
            self.sum_phi_L[name] += vec
        # per-sample storage for L (broadcast) so MI can be computed later
        if self.store_samples:
            self._maybe_store_sample(name, 'L', vec)

    def update_y_ref(self, per_module_y: Dict[str, torch.Tensor]):
        if self.yref_mode != "running_avg":
            return
        for name, y_vec in per_module_y.items():
            y = y_vec.detach().cpu().float()
            if name not in self.y_ref:
                self.y_ref[name] = y
                self.y_ref_count[name] = 1
            else:
                c = self.y_ref_count[name]
                self.y_ref[name] = (self.y_ref[name] * c + y) / (c + 1)
                self.y_ref_count[name] = c + 1

    def finalize_dataset(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return averaged phi over inputs: {module: {phi_a, phi_g, phi_L}}"""
        if self.n_inputs <= 0:
            return {}
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in set(list(self.sum_phi_a.keys()) + list(self.sum_phi_g.keys()) + list(self.sum_phi_L.keys())):
            out[name] = {
                "phi_a": self.sum_phi_a[name] / self.n_inputs if name in self.sum_phi_a else None,
                "phi_g": self.sum_phi_g[name] / self.n_inputs if name in self.sum_phi_g else None,
                "phi_L": self.sum_phi_L[name] / self.n_inputs if name in self.sum_phi_L else None,
            }
            # attach per-sample tensors if requested
            if self.store_samples and self.samples_phi_a.get(name):
                try:
                    out[name]['phi_a_samples'] = torch.stack(self.samples_phi_a[name], dim=0)
                except Exception:
                    pass
            if self.store_samples and self.samples_phi_g.get(name):
                try:
                    out[name]['phi_g_samples'] = torch.stack(self.samples_phi_g[name], dim=0)
                except Exception:
                    pass
            if self.store_samples and self.samples_phi_L.get(name):
                try:
                    out[name]['phi_L_samples'] = torch.stack(self.samples_phi_L[name], dim=0)
                except Exception:
                    pass
        return out

    # ----------------- internal helpers -----------------
    def _maybe_store_sample(self, name: str, kind: str, vec: torch.Tensor):
        if not self.store_samples:
            return
        lst_map = {
            'a': self.samples_phi_a,
            'g': self.samples_phi_g,
            'L': self.samples_phi_L,
        }
        if kind not in lst_map:
            return
        self._samples_attempts[name] += 1
        max_s = self.mi_max_samples
        if max_s == 0:
            return  # explicitly disable storage
        if max_s > 0 and len(lst_map[kind][name]) >= max_s:
            return  # simple cap (could extend to reservoir)
        lst_map[kind][name].append(vec.detach().cpu().float().clone())


def get_lape_hook(name: str, tracker: _LAPETracker, req_act: Iterable[str]):
    req = set(req_act)
    def hook_fn(module, args, kwargs, output):
        # convert tensors to [B, H]
        def _to_bh(t: torch.Tensor) -> Optional[torch.Tensor]:
            try:
                if t is None or not isinstance(t, torch.Tensor):
                    return None
                tt = t.detach().to("cpu", non_blocking=True).float()
                if tt.dim() == 3:
                    return tt.sum(dim=1)
                if tt.dim() == 2:
                    return tt
                if tt.dim() == 1:
                    return tt.unsqueeze(0)
                h = tt.shape[-1]
                return tt.reshape(-1, h)
            except Exception:
                return None

        out_tensor = output[0] if isinstance(output, tuple) else output
        out_bh = _to_bh(out_tensor) if isinstance(out_tensor, torch.Tensor) and ("output" in req) else None
        in_tensor = kwargs.get("hidden_states", args[0] if (args and isinstance(args[0], torch.Tensor)) else None)
        in_bh = _to_bh(in_tensor) if ("input" in req) and isinstance(in_tensor, torch.Tensor) else None
        tracker.hook_record(name, in_bh, out_bh)
    return hook_fn


# -------------------------
# Main caching routine
# -------------------------

@torch.no_grad()
def main():
    args = parse_args()

    # Apply GPU selection early
    if getattr(args, "gpus", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"[Env] CUDA_VISIBLE_DEVICES set to: {args.gpus}")

    # If CUDA is not available but user kept --llm-device=cuda, fallback to cpu to avoid runtime error
    if getattr(args, "llm_device", "cuda") == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available; falling back to --llm-device=cpu")
        args.llm_device = "cpu"

    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)
    tmp_img_dir = osp.join(work_dir, 'tmp_images')
    os.makedirs(tmp_img_dir, exist_ok=True)

    # -------- Determine mode: VLMEval VLM vs transformers LLM --------
    use_hf_llm = args.hf_llm_id is not None and len(args.hf_llm_id) > 0
    if not use_hf_llm and not args.model:
        raise ValueError("Please provide either --model (VLMEvalKit) or --hf-llm-id (transformers).")

    # -------- Build dataset payload --------
    dataset_id: str
    messages: List[List[Dict[str, Any]]] = []  # for VLM path (multi-modal messages)
    texts_for_llm: List[str] = []              # for LLM path (plain text)

    if args.hf_dataset is not None:
        key = args.hf_dataset.strip().lower()
        if key != 'meta':
            raise ValueError("--hf-dataset currently only supports 'meta'")
        if load_dataset is None:
            raise RuntimeError(
                f"datasets is not available; please install `datasets`. Import error: {_DATASETS_IMPORT_ERR}"
            )
        samples = build_meta_probe_dataset(args)
        if use_hf_llm:
            # New: text-only prompts for transformers LLM
            texts_for_llm = _get_texts_from_meta_samples(samples)
        else:
            for s in samples:
                img = s["image"]
                # convert + save
                if Image is not None and isinstance(img, Image.Image) and img.mode == 'RGBA':
                    img = img.convert('RGB')
                img_path = _dump_image_to_file(img, tmp_img_dir)
                text = s.get("question", "")
                messages.append([
                    dict(type='image', value=img_path),
                    dict(type='text', value=text)
                ])
        dataset_id = 'HF:meta'
    elif args.data is not None:
        if vlmeval_build_dataset is None:
            raise RuntimeError("vlmeval.dataset.build_dataset is unavailable; cannot load --data dataset.")
        # VLMEvalKit build_dataset has had signature changes across versions.
        # Try with 'dataset_name' first, then fallback to 'name', then positional.
        try:
            dataset = vlmeval_build_dataset(dataset_name=args.data)
        except TypeError:
            try:
                dataset = vlmeval_build_dataset(name=args.data)
            except TypeError:
                dataset = vlmeval_build_dataset(args.data)
        dataset_id = dataset.dataset_name
        if use_hf_llm:
            texts_for_llm = _extract_texts_from_vlmeval_dataset(dataset, args.max_samples)
        else:
            data = dataset.data
            max_n = len(data) if args.max_samples is None else min(len(data), args.max_samples)
            for i in range(max_n):
                struct = dataset.build_prompt(data.iloc[i])
                messages.append(struct)
    else:
        raise ValueError("Please specify either --hf-dataset meta or --data <DatasetName>.")

    # -------- Instantiate model (either VLMEvalKit VLM or transformers LLM) --------
    torch_model: Optional[nn.Module] = None
    vlm = None
    if use_hf_llm:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError("transformers is not available; please install transformers to use --hf-llm-id")

        # parse dtype
        dt = args.llm_dtype
        torch_dtype = None
        if dt == "float16":
            torch_dtype = torch.float16
        elif dt == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dt == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = None  # auto

        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_llm_id, use_fast=True, trust_remote_code=args.trust_remote_code
        )
        # Ensure pad token exists for batching (common for LLaMA)
        if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        device_map = args.llm_device_map
        if device_map is not None and device_map.lower() == "none":
            device_map = None

        if device_map is not None:
            # Multi-GPU sharded load via accelerate
            model = AutoModelForCausalLM.from_pretrained(
                args.hf_llm_id,
                torch_dtype=torch_dtype,
                trust_remote_code=args.trust_remote_code,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.hf_llm_id,
                torch_dtype=torch_dtype,
                trust_remote_code=args.trust_remote_code,
            )
            device = torch.device(args.llm_device)
            model.to(device)
        model.eval()
        torch_model = model
    else:
        # VLMEvalKit (safe WORLD_SIZE handling)
        ws_bak = os.environ.pop('WORLD_SIZE', None)
        model_kwargs = {}
        model_name = args.model
        # Pass through optional vLLM usage for specific models
        if model_name is not None and (
            'Llama-4' in model_name or 'Qwen2-VL' in model_name or 'Qwen2.5-VL' in model_name
        ):
            model_kwargs['use_vllm'] = args.use_vllm
        # New: propagate --vlm-device-map to wrapper so it can avoid eager full .cuda() (especially for LLaVA)
        if getattr(args, 'vlm_device_map', None):
            model_kwargs['device_map'] = args.vlm_device_map

        constructor = supported_VLM[model_name]
        if isinstance(constructor, functools.partial):
            kw = dict(constructor.keywords or {})
            if 'model_path' in kw and isinstance(kw['model_path'], str):
                kw['model_path'] = kw['model_path'].strip()
            constructor = functools.partial(constructor.func, *(constructor.args or ()), **kw)
        vlm = constructor(**model_kwargs) if 'constructor' in locals() else supported_VLM[model_name](**model_kwargs)
        if ws_bak:
            os.environ['WORLD_SIZE'] = ws_bak

        if getattr(vlm, 'is_api', False):
            raise RuntimeError("API models are not supported for activation caching (no torch hooks).")

        # If we used a VLMEval dataset (not HF meta), let model know how to dump images if needed
        if args.hf_dataset is None and args.data is not None and hasattr(vlm, 'set_dump_image') and hasattr(dataset, 'dump_image'):
            vlm.set_dump_image(dataset.dump_image)

        torch_model = get_underlying_torch_model(vlm)
        if torch_model is None:
            raise RuntimeError("Cannot find underlying torch model to hook (no .model and wrapper is not nn.Module)")
        torch_model.eval()

        # Some VLMEvalKit wrappers (e.g., mPLUG-Owl2) retain constructor kwargs and pass them
        # directly into model.generate(). The HF generate() API does NOT accept 'device_map',
        # so if we passed --vlm-device-map earlier it will raise:
        #   ValueError: The following `model_kwargs` are not used by the model: ['device_map']
        # The model has already been loaded (and optionally sharded below via accelerate),
        # so we safely remove this key from generation kwargs to avoid the runtime error.
        try:
            if hasattr(vlm, 'kwargs') and isinstance(getattr(vlm, 'kwargs'), dict) and 'device_map' in vlm.kwargs:
                _dm_val = vlm.kwargs.pop('device_map', None)
                if args.verbose:
                    print(f"[VLM] Stripped unsupported 'device_map'={_dm_val} from vlm.kwargs before generation.")
        except Exception:
            pass

        # Suppress repeated HF warnings about missing pad_token_id by setting it to eos_token_id
        try:
            if hasattr(torch_model, 'config'):
                cfg = torch_model.config
                if getattr(cfg, 'pad_token_id', None) is None and getattr(cfg, 'eos_token_id', None) is not None:
                    cfg.pad_token_id = cfg.eos_token_id
                    if args.verbose:
                        print(f"[VLM] Set config.pad_token_id to eos_token_id={cfg.eos_token_id}.")
                # Also propagate to generation kwargs so wrapper passes it explicitly
                if hasattr(vlm, 'kwargs') and isinstance(getattr(vlm, 'kwargs'), dict) and 'pad_token_id' not in vlm.kwargs and getattr(cfg, 'pad_token_id', None) is not None:
                    vlm.kwargs['pad_token_id'] = cfg.pad_token_id
                    if args.verbose:
                        print(f"[VLM] Injected pad_token_id={cfg.pad_token_id} into vlm.kwargs.")
        except Exception:
            pass

        # Optional: shard the underlying HF model across multiple GPUs using accelerate
        if getattr(args, "vlm_device_map", None):
            if _acc_dispatch_model is None:
                warnings.warn("accelerate is not available; --vlm-device-map ignored. Please 'pip install accelerate'.")
            else:
                # resolve dtype
                _dt = (args.vlm_dtype or "auto").lower()
                _torch_dtype = None
                if _dt == "float16":
                    _torch_dtype = torch.float16
                elif _dt == "bfloat16":
                    _torch_dtype = torch.bfloat16
                elif _dt == "float32":
                    _torch_dtype = torch.float32

                try:
                    dmap = args.vlm_device_map
                    if isinstance(dmap, str) and dmap.lower() == "auto":
                        if _acc_get_balanced_memory is None or _acc_infer_auto_device_map is None:
                            warnings.warn("accelerate auto device map utilities unavailable; skip sharding.")
                        else:
                            # Move model to CPU first to avoid double allocation on GPU during dispatch
                            try:
                                any_cuda = any(p.is_cuda for p in torch_model.parameters())
                            except Exception:
                                any_cuda = False
                            if any_cuda:
                                if args.verbose:
                                    print("[VLM][accelerate] Moving model to CPU before dispatch to avoid GPU OOM...")
                                torch_model.to('cpu')
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            no_split = list(getattr(args, "vlm_no_split_classes", []) or [])
                            # Balanced memory if available
                            try:
                                max_mem = _acc_get_balanced_memory(torch_model, dtype=_torch_dtype, no_split_module_classes=no_split) if _acc_get_balanced_memory else None
                            except Exception:
                                max_mem = None
                            # Infer device map
                            if max_mem is not None:
                                inferred = _acc_infer_auto_device_map(torch_model, max_memory=max_mem, no_split_module_classes=no_split)
                            else:
                                inferred = _acc_infer_auto_device_map(torch_model, no_split_module_classes=no_split)
                            _acc_dispatch_model(torch_model, device_map=inferred)
                            if args.verbose:
                                print("[VLM][accelerate] Dispatched underlying model with auto device map:")
                                try:
                                    print(inferred)
                                except Exception:
                                    pass
                                # Print parameter device distribution
                                try:
                                    from collections import Counter as _Counter
                                    cnt = _Counter(str(p.device) for p in torch_model.parameters() if hasattr(p, 'device'))
                                    print(f"[VLM][accelerate] Param device counts: {dict(cnt)}")
                                except Exception:
                                    pass
                    else:
                        warnings.warn("Custom --vlm-device-map is not parsed in this script; only 'auto' is supported.")
                except Exception as e:
                    warnings.warn(f"[VLM][accelerate] dispatch failed: {type(e).__name__}: {e}")

    target_modules = get_target_module_map(
        torch_model,
        module_regex=args.module_regex,
        include_types=args.include_types or [],
        exclude_regex=args.exclude_regex,
        verbose=args.verbose,
    )
    if not target_modules:
        raise RuntimeError("No modules matched given --module-regex/--include-types/--exclude-regex filters.")

    activation_stats = defaultdict(lambda: {
        "input_sum": None, "input_tokens": 0,
        "output_sum": None, "output_tokens": 0
    })

    keep_batches = bool(getattr(args, "fai_compute", False))
    hooks = [
        module.register_forward_hook(
            get_hook_with_kwargs(name, args.req_act, activation_stats, keep_batches=keep_batches), with_kwargs=True
        )
        for name, module in target_modules.items()
    ]

    # LAPE: separate hook set and tracker (works for LLM and VLM)
    lape_tracker: Optional[_LAPETracker] = None
    lape_hooks = []
    if getattr(args, "lape_enable", False):
        lape_tracker = _LAPETracker(
            list(target_modules.keys()),
            args.req_act,
            args.lape_gamma,
            args.lape_yref,
            store_samples=getattr(args, 'lape_mi_store', False),
            mi_max_samples=getattr(args, 'lape_mi_max_samples', 512),
        )
        lape_hooks = [
            module.register_forward_hook(get_lape_hook(name, lape_tracker, args.req_act), with_kwargs=True)
            for name, module in target_modules.items()
        ]

    # -------- Run forwards to collect activations --------
    processed = 0
    err_count = 0
    if use_hf_llm:
        # New: batch text-only forwards with transformers
        assert torch_model is not None
        model = torch_model  # type: ignore
        device_map = args.llm_device_map
        device = torch.device(args.llm_device)
        # tokenizer defined above in LLM branch
        # To avoid mypy confusion, re-create here (safe, loads from cache)
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_llm_id, use_fast=True, trust_remote_code=args.trust_remote_code
        )
        if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        bsz = max(1, int(args.probe_batch_size))
        for i in tqdm(range(0, len(texts_for_llm), bsz), desc=f"Forward LLM on {dataset_id}"):
            batch_texts = texts_for_llm[i:i+bsz]
            # When LAPE disabled: single forward per sample (original behavior, vectorized by batch)
            if not getattr(args, "lape_enable", False):
                enc = tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=args.llm_max_length
                )
                target_dev = _pick_llm_input_device(model) if device_map is not None and not (isinstance(device_map, str) and device_map.lower() == "none") else device
                enc = {k: v.to(target_dev) for k, v in enc.items()}
                try:
                    if args.llm_forward_mode == "generate":
                        _ = model.generate(
                            input_ids=enc.get("input_ids"),
                            attention_mask=enc.get("attention_mask"),
                            max_new_tokens=max(1, int(getattr(args, "llm_new_tokens", 1))),
                            do_sample=False,
                            temperature=0.0,
                            use_cache=True,
                        )
                    else:
                        _ = model(**enc, use_cache=False)
                except RuntimeError as err:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if os.environ.get('SKIP_ERR', '0') == '1':
                        warnings.warn(f"forward failed: {type(err).__name__}: {err}")
                        err_count += 1
                    else:
                        raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                processed += len(batch_texts)
                continue

            # LAPE enabled: process each sample independently to run multi-path sampling
            for text in batch_texts:
                enc_single = tokenizer(
                    [text],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=args.llm_max_length
                )
                target_dev = _pick_llm_input_device(model) if device_map is not None and not (isinstance(device_map, str) and device_map.lower() == "none") else device
                enc_single = {k: v.to(target_dev) for k, v in enc_single.items()}

                path_records: List[Tuple[float, int, Dict[str, Dict[str, torch.Tensor]]]] = []
                Ns = max(1, int(args.lape_samples))
                # propagate unified sampling temperature via env for wrappers that honor it
                os.environ['LAPE_TEMPERATURE'] = str(max(float(getattr(args, 'lape_temperature', 0.7)), 1e-5))
                fixed_new_len = int(args.lape_max_new) if getattr(args, 'lape_deterministic', False) else None
                for k in range(Ns):
                    # randomize new tokens length in [min,max]
                    if fixed_new_len is not None:
                        new_len = fixed_new_len
                        top_p = 1.0
                        temp = 0.0
                    else:
                        Lmin = max(1, int(args.lape_min_new))
                        Lmax = max(Lmin, int(args.lape_max_new))
                        new_len = int(np.random.randint(Lmin, Lmax + 1))
                        top_p = float(args.lape_top_p)
                        temp = max(float(getattr(args, 'lape_temperature', 0.7)), 1e-5)
                    lape_tracker.start_path()  # reset buffers for this path
                    try:
                        gen_out = model.generate(
                            input_ids=enc_single.get("input_ids"),
                            attention_mask=enc_single.get("attention_mask"),
                            max_new_tokens=new_len,
                            do_sample=not getattr(args, 'lape_deterministic', False),
                            top_p=top_p,
                            temperature=temp,
                            use_cache=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    except RuntimeError as err:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        if os.environ.get('SKIP_ERR', '0') == '1':
                            warnings.warn(f"LAPE path generate failed: {type(err).__name__}: {err}")
                            err_count += 1
                            continue
                        else:
                            raise

                    # compute log prob of sampled sequence and length-normalized loss
                    scores = gen_out.scores  # list of length new_len, each [B=1, V]
                    if not scores:
                        continue
                    # generated tokens are the last new_len tokens of sequences
                    seq = gen_out.sequences  # [1, src_len + new_len]
                    gen_token_ids = seq[:, -len(scores):]  # [1, new_len]
                    log_p_sum = 0.0
                    L_norm_sum = 0.0
                    for t, logits in enumerate(scores):
                        # logits shape [1, V]
                        logp = F.log_softmax(logits, dim=-1)  # [1, V]
                        tok_id = gen_token_ids[:, t]
                        tok_logp = logp.gather(-1, tok_id.unsqueeze(-1)).squeeze(-1)  # [1]
                        log_p_sum += float(tok_logp.item())
                        L_norm_sum += float((-tok_logp).item())
                    # length-normalized loss (average per generated token)
                    L_norm = L_norm_sum / max(len(scores), 1)

                    # finalize per-path module vectors from tracker buffers
                    vecs = lape_tracker._finalize_single_path_vectors()
                    # update running y_ref with current path outputs
                    y_latest = {n: v["y"] for n, v in vecs.items()}
                    lape_tracker.update_y_ref(y_latest)
                    # store record: (log p, length, per-module vectors)
                    path_records.append((log_p_sum, len(scores), vecs))

                # combine Ns paths for this input
                lape_tracker.accumulate_one_input(path_records)
                # add scalar L contribution per module using same normalized weights
                if path_records:
                    logws = []
                    for (log_p, Lk, _) in path_records:
                        logw = float(log_p) + float(max(Lk - 1, 0)) * float(np.log(max(args.lape_gamma, 1e-8)))
                        logws.append(logw)
                    logws_t = torch.tensor(logws, dtype=torch.float32)
                    ws = torch.softmax(logws_t, dim=0).tolist()
                    # weighted average of L_norm scalars
                    # recompute per-path L_norm since we didn't retain above; compute again quickly from stored data
                    # For efficiency, we approximated L_norm by using stored L_norm_sum/len at time, so reuse via recompute
                    # Here we simply approximate by setting same average across modules (since it's scalar)
                    # Compute weighted scalar from stored path_records by re-walking outputs
                    # We'll re-evaluate using the length-normalized negative log-prob only; scores no longer available, so approximate via exp(log_p/len)
                    # Fallback: use exp-based approx to derive average -log p per token
                    L_scalars = []
                    for (log_p, Lk, _vecs) in path_records:
                        # approx: mean NLL per token = -(log_p) / Lk
                        L_scalars.append(float(-log_p) / max(Lk, 1))
                    L_weighted = float(np.dot(ws, L_scalars))
                    # Broadcast to all modules present in combined vecs
                    mod_names = set()
                    for _, _, vecs in path_records:
                        mod_names.update(vecs.keys())
                    for name in mod_names:
                        # Use H from phi_a accumulator if exists; otherwise from current vecs
                        if name in lape_tracker.sum_phi_a:
                            H = int(lape_tracker.sum_phi_a[name].shape[0])
                        else:
                            # find any vec to get H
                            H = None
                            for _, _, vecs in path_records:
                                if name in vecs:
                                    H = int(vecs[name]["a"].shape[0])
                                    break
                            if H is None:
                                continue
                        lape_tracker.accumulate_L_scalar(name, L_weighted, H)

                processed += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        # VLM path
        if not getattr(args, "lape_enable", False):
            # Original: single forward per message
            for struct in tqdm(messages, desc=f"Forward {args.model} on {dataset_id}"):
                if os.environ.get('SKIP_ERR', '0') == '1':
                    try:
                        _ = vlm.generate(message=struct, dataset=dataset_id)
                    except RuntimeError as err:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        warnings.warn(f"generation failed: {type(err).__name__}: {err}")
                        err_count += 1
                else:
                    _ = vlm.generate(message=struct, dataset=dataset_id)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                processed += 1
        else:
            # LAPE enabled: multi-path sampling per input via wrapper
            Ns = max(1, int(args.lape_samples))
            # Try to tweak wrapper kwargs if present
            has_kwargs = hasattr(vlm, 'kwargs') and isinstance(getattr(vlm, 'kwargs'), dict)
            for struct in tqdm(messages, desc=f"LAPE sample {args.model} on {dataset_id}"):
                path_records: List[Tuple[float, int, Dict[str, Dict[str, torch.Tensor]]]] = []
                # backup and set sampling flags
                bak = None
                if has_kwargs:
                    bak = dict(vlm.kwargs)
                # Use environment flag to hint wrappers to capture scores
                os.environ['LAPE_CAPTURE'] = '1'
                fixed_new_len = int(args.lape_max_new) if getattr(args, 'lape_deterministic', False) else None
                for k in range(Ns):
                    try:
                        if has_kwargs:
                            # enable sampling and unify parameters
                            if getattr(args, 'lape_deterministic', False):
                                vlm.kwargs['do_sample'] = False
                                vlm.kwargs['top_p'] = 1.0
                                vlm.kwargs['temperature'] = 0.0
                                new_len = fixed_new_len
                            else:
                                vlm.kwargs['do_sample'] = True
                                vlm.kwargs['top_p'] = float(args.lape_top_p)
                                vlm.kwargs['temperature'] = max(float(getattr(args, 'lape_temperature', 0.7)), 1e-5)
                                Lmin = max(1, int(args.lape_min_new))
                                Lmax = max(Lmin, int(args.lape_max_new))
                                new_len = int(np.random.randint(Lmin, Lmax + 1))
                            vlm.kwargs['max_new_tokens'] = new_len
                            # do not set return_dict/output_scores here to avoid global config warnings;
                            # wrappers will enable them when LAPE_CAPTURE=1
                        # reset per-path buffers
                        if lape_tracker is not None:
                            lape_tracker.start_path()
                        _ = vlm.generate(message=struct, dataset=dataset_id)
                        # attempt to fetch logp and len from wrapper if available
                        logp = None
                        Lk = None
                        if hasattr(vlm, '_last_lape_logp') and hasattr(vlm, '_last_lape_len'):
                            try:
                                logp = float(getattr(vlm, '_last_lape_logp'))
                                Lk = int(getattr(vlm, '_last_lape_len'))
                            except Exception:
                                logp = None; Lk = None
                        # finalize vectors for this path
                        if lape_tracker is not None:
                            vecs = lape_tracker._finalize_single_path_vectors()
                            # update y_ref with current outputs
                            lape_tracker.update_y_ref({n: v['y'] for n, v in vecs.items()})
                            if logp is None or Lk is None:
                                # fallback: equal weights; set logp=0, Lk=new_len if known
                                if has_kwargs:
                                    Lk = int(vlm.kwargs.get('max_new_tokens', 1))
                                else:
                                    Lk = 1
                                logp = 0.0
                            path_records.append((logp, Lk, vecs))
                    except RuntimeError as err:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        if os.environ.get('SKIP_ERR', '0') == '1':
                            warnings.warn(f"VLM LAPE generate failed: {type(err).__name__}: {err}")
                            err_count += 1
                            continue
                        else:
                            raise
                # restore env/kwargs
                os.environ.pop('LAPE_CAPTURE', None)
                os.environ.pop('LAPE_TEMPERATURE', None)
                if has_kwargs and bak is not None:
                    vlm.kwargs = bak

                # accumulate this input
                if lape_tracker is not None:
                    lape_tracker.accumulate_one_input(path_records)
                    # compute weighted L scalar across paths
                    if path_records:
                        logws = []
                        for (log_p, Lk, _) in path_records:
                            logw = float(log_p) + float(max(Lk - 1, 0)) * float(np.log(max(args.lape_gamma, 1e-8)))
                            logws.append(logw)
                        ws = torch.softmax(torch.tensor(logws, dtype=torch.float32), dim=0).tolist()
                        L_scalars = [float(-lp) / max(Lk, 1) for (lp, Lk, _) in path_records]
                        L_weighted = float(np.dot(ws, L_scalars))
                        mod_names = set()
                        for _, _, vecs in path_records:
                            mod_names.update(vecs.keys())
                        for name in mod_names:
                            # determine H
                            H = None
                            if name in lape_tracker.sum_phi_a:
                                H = int(lape_tracker.sum_phi_a[name].shape[0])
                            else:
                                for _, _, vecs in path_records:
                                    if name in vecs:
                                        H = int(vecs[name]['a'].shape[0]); break
                            if H is not None:
                                lape_tracker.accumulate_L_scalar(name, L_weighted, H)

                processed += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    for h in hooks:
        h.remove()
    for h in lape_hooks:
        try:
            h.remove()
        except Exception:
            pass

    if args.verbose:
        print(f"Processed samples: {processed}, errors: {err_count}")

    # -------- Aggregate & compute FAI (optional) --------
    fai_results: Dict[str, Dict[str, torch.Tensor]] = {}
    if getattr(args, "fai_compute", False):
        if "input" not in args.req_act:
            warnings.warn("FAI requested but input activations were not recorded. Will approximate MI using output signals only.")

        # helper: cap samples per module
        def _cap_cat(batches: List[torch.Tensor], cap: int) -> Optional[torch.Tensor]:
            if not batches:
                return None
            mat = torch.cat(batches, dim=0)
            if cap > 0 and mat.shape[0] > cap:
                # uniform subsample rows
                idx = torch.randperm(mat.shape[0])[:cap]
                mat = mat[idx]
            return mat

        # MI backends
        _have_sklearn = False
        try:
            from sklearn.feature_selection import mutual_info_regression as _mi_reg  # type: ignore
            _have_sklearn = True
        except Exception:
            _mi_reg = None  # type: ignore

        def _rankdata(x: np.ndarray) -> np.ndarray:
            # simple average-rank implementation
            order = x.argsort()
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(x) + 1, dtype=float)
            return ranks

        def _mi_hist(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
            # Histogram-based MI approximation; robust fallback w/o sklearn
            if x.ndim != 1:
                x = x.reshape(-1)
            if y.ndim != 1:
                y = y.reshape(-1)
            # Protect against degenerate arrays
            if x.size < 5 or np.allclose(x.var(), 0.0) or y.size < 5 or np.allclose(y.var(), 0.0):
                return 0.0
            try:
                c_xy, _, _ = np.histogram2d(x, y, bins=bins)
                p_xy = c_xy / np.maximum(c_xy.sum(), 1.0)
                p_x = p_xy.sum(axis=1, keepdims=True)
                p_y = p_xy.sum(axis=0, keepdims=True)
                nz = p_xy > 0
                mi = float((p_xy[nz] * (np.log(p_xy[nz]) - np.log(p_x[nz.any(axis=1), 0][:, None]) - np.log(p_y[0, nz.any(axis=0)][None, :]))).sum())
                return max(mi, 0.0)
            except Exception:
                return 0.0

        def _pearson_abs(x: np.ndarray, y: np.ndarray) -> float:
            if x.ndim != 1:
                x = x.reshape(-1)
            if y.ndim != 1:
                y = y.reshape(-1)
            if x.size < 2 or y.size < 2:
                return 0.0
            xv = x - x.mean(); yv = y - y.mean()
            den = (np.sqrt((xv**2).sum()) * np.sqrt((yv**2).sum()))
            if den <= 0:
                return 0.0
            return float(abs((xv * yv).sum() / den))

        def _spearman_abs(x: np.ndarray, y: np.ndarray) -> float:
            rx = _rankdata(x); ry = _rankdata(y)
            return _pearson_abs(rx, ry)

        def _compute_mi_vector(x_signal: np.ndarray, Y: np.ndarray, mode: str) -> np.ndarray:
            # x_signal shape [N], Y shape [N, H]
            N, H = Y.shape
            mi = np.zeros(H, dtype=np.float64)
            if mode == "mi" and _have_sklearn:
                try:
                    # sklearn expects X: [N, d]; y: [N]
                    X = x_signal.reshape(-1, 1)
                    for i in range(H):
                        val = _mi_reg(X, Y[:, i], discrete_features=False)
                        # mutual_info_regression can return 0-d np arrays; extract scalar explicitly
                        try:
                            mi[i] = float(val.item())  # type: ignore[attr-defined]
                        except Exception:
                            mi[i] = float(np.asarray(val))
                    return mi
                except Exception:
                    pass  # fallback below
            # fallback: histogram MI or correlations
            if mode == "pearson":
                for i in range(H):
                    mi[i] = _pearson_abs(x_signal, Y[:, i])
            elif mode == "spearman":
                for i in range(H):
                    mi[i] = _spearman_abs(x_signal, Y[:, i])
            else:
                for i in range(H):
                    mi[i] = _mi_hist(x_signal, Y[:, i])
            return mi

        cap = max(0, int(getattr(args, "fai_max_samples_per_module", 4096)))
        eps = float(getattr(args, "fai_eps", 1e-6))
        fai_mode = str(getattr(args, "fai_mi_mode", "mi"))

        # Progress over modules for FAI computation
        _mods = list(activation_stats.items())
        for name, stats in tqdm(_mods, desc="Compute FAI", total=len(_mods)):
            X_mat = _cap_cat(stats.get("input_batches", []), cap)
            Y_mat = _cap_cat(stats.get("output_batches", []), cap)
            if Y_mat is None or Y_mat.numel() == 0:
                continue
            Y_np = Y_mat.numpy()
            # input signal: prefer L2 norm of input rows; else use L2 norm of output rows
            if X_mat is not None and X_mat.numel() > 0:
                x_signal = torch.linalg.norm(X_mat, dim=1).numpy()
            else:
                x_signal = torch.linalg.norm(Y_mat, dim=1).numpy()
            try:
                mi_vec = _compute_mi_vector(x_signal, Y_np, fai_mode)
            except Exception:
                mi_vec = np.zeros(Y_np.shape[1], dtype=np.float64)
            # A_i: mean absolute output activation per neuron
            A_vec = np.mean(np.abs(Y_np), axis=0)
            # Hessian diag proxy: variance of outputs per neuron
            H_diag = np.var(Y_np, axis=0)
            fai = (A_vec * mi_vec) / (H_diag + eps)
            fai_results[name] = {
                "fai": torch.from_numpy(fai.astype(np.float32)),
                "A": torch.from_numpy(A_vec.astype(np.float32)),
                "MI": torch.from_numpy(mi_vec.astype(np.float32)),
                "H_proxy": torch.from_numpy(H_diag.astype(np.float32)),
                "n_samples": torch.tensor(int(Y_np.shape[0])),
            }

    # -------- Aggregate & save (avg activations + optional FAI + optional LAPE) --------
    averaged_activations: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, stats in activation_stats.items():
        averaged_activations[name] = {}
        if stats["input_sum"] is not None and stats["input_tokens"] > 0:
            averaged_activations[name]["input"] = stats["input_sum"] / stats["input_tokens"]
        if stats["output_sum"] is not None and stats["output_tokens"] > 0:
            averaged_activations[name]["output"] = stats["output_sum"] / stats["output_tokens"]
        # attach FAI vectors if computed for this module
        if getattr(args, "fai_compute", False) and name in fai_results:
            averaged_activations[name]["fai"] = fai_results[name]["fai"]
            averaged_activations[name]["fai_A"] = fai_results[name]["A"]
            averaged_activations[name]["fai_MI"] = fai_results[name]["MI"]
            averaged_activations[name]["fai_H"] = fai_results[name]["H_proxy"]
            averaged_activations[name]["fai_n_samples"] = fai_results[name]["n_samples"]

    lape_out: Dict[str, Dict[str, torch.Tensor]] = {}
    if getattr(args, "lape_enable", False) and lape_tracker is not None:
        lres = lape_tracker.finalize_dataset()
        # Rename keys to concise names in save file
        for name, dd in lres.items():
            lape_out[name] = {}
            if dd.get("phi_a") is not None:
                lape_out[name]["phi_a"] = dd["phi_a"]
            if dd.get("phi_g") is not None:
                lape_out[name]["phi_g"] = dd["phi_g"]
            if dd.get("phi_L") is not None:
                lape_out[name]["phi_L"] = dd["phi_L"]

    save_path = args.save
    if not save_path:
        os.makedirs('activations', exist_ok=True)
        suffix = (args.hf_dataset or args.data or 'unknown').replace('/', '_')
        model_name_ = args.hf_llm_id if use_hf_llm else args.model
        # 若传入的是绝对路径或包含斜杠的 HF 模型 ID，使用 basename，避免 os.path.join 被绝对路径覆盖
        if isinstance(model_name_, str):
            model_label = osp.basename(model_name_.rstrip(os.sep))
        else:
            model_label = str(model_name_)
        save_path = osp.join('activations', f"{model_label}_{suffix}.pt")
    # Save averaged activations
    torch.save(averaged_activations, save_path)
    # Save LAPE results if available, as a sidecar file
    if lape_out:
        base, ext = osp.splitext(save_path)
        lape_path = base + "_lape" + ext
        torch.save(lape_out, lape_path)
        print(f"[Done] Saved LAPE (phi_a/phi_g/phi_L) to: {lape_path}")
    else:
        if getattr(args, "lape_enable", False):
            # Provide a helpful hint when LAPE is enabled but nothing was saved
            try:
                n_inputs = int(lape_tracker.n_inputs) if lape_tracker is not None else 0  # type: ignore[attr-defined]
            except Exception:
                n_inputs = 0
            warnings.warn(
                f"LAPE was enabled but no LAPE vectors were saved (empty results). "
                f"Processed inputs for LAPE: {n_inputs}. "
                f"This can happen if generation failed for all samples, hooks didn't fire, or module filters matched none."
            )

    # Cleanup
    del hooks, activation_stats, averaged_activations
    del lape_hooks
    if not use_hf_llm:
        del vlm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Done] Saved averaged activations to: {save_path}")


if __name__ == "__main__":
    main()
