#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSF-TEFM Stage-1: Subspace Extraction (SVD based)

目标:
  对单个模型 (base 或 donor) 的可合并线性层权重执行截断 SVD (top-r)，
  生成用于后续 Gromov-Wasserstein 子空间对齐的点云表示 (奇异向量集合)。

输出 (torch.save .pt): dict[module_name] = {
    'U': FloatTensor[d_out, r],
    'S': FloatTensor[r],
    'rank_used': int,
    'param_key': 原始参数键名 (weight),
    'shape': (d_out, d_in)
}

设计说明:
  - 仅处理需要合并的线性层 (utils.need_merge) 且维度为 2D、以 .weight 结尾。
  - rank r 会被截断为 min(r, d_out, d_in)。
  - 默认使用 torch.linalg.svd / svd_lowrank；当 d_out 与 d_in 都较大且 r 远小于最小维度时尝试 lowrank 路径。
  - 不保存 V 以节省存储；后续 Stage-3 只需要左奇异向量子空间即可。

CLI 示例:
  python merge/gsf/gsf_stage1_subspace.py \
      --model-dir downloaded_models/llava-v1.5-7b/Llama-2-7b-hf \
      --rank 64 --save activations/llava_subspace.pt

依赖: merge/gsf/utils.py (load_weights, need_merge)
"""
from __future__ import annotations
import argparse
import math
import os
import os.path as osp
from typing import Dict

import torch

from utils import load_weights, need_merge  # type: ignore


def _canon_module_from_param(key: str) -> str:
    # 去掉最终 .weight/.bias，并规范 language_model 前缀
    k = key.replace("language_model.model.", "model.").replace("language_model.", "model.")
    if "layers" in k:
        pos = k.find("layers")
        k = "model." + k[pos:]
    parts = k.split('.')
    if parts[-1] in ('weight', 'bias'):
        parts = parts[:-1]
    return '.'.join(parts)


def extract_subspaces(args: argparse.Namespace):
    print("\n--- [GSF Stage-1: Subspace SVD Extraction] ---")
    weights = load_weights(args.model_dir)
    rank = int(args.rank)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    total = 0
    kept = 0
    for k, W in weights.items():
        if not k.endswith('.weight'):
            continue
        if not need_merge(k):
            continue
        if W.ndim != 2:
            continue
        total += 1
        Wf = W.float()
        d_out, d_in = Wf.shape
        r_use = min(rank, d_out, d_in)
        if r_use <= 0:
            continue
        # 选择算法: 若 r_use << min(d_out,d_in) 且 2 * r_use < min(d_out,d_in) 尝试 lowrank
        try_lowrank = (hasattr(torch.linalg, 'svd') and r_use < min(d_out, d_in) // 2)
        try:
            if try_lowrank and hasattr(torch, 'svd_lowrank'):
                # torch.svd_lowrank 在一些版本位于 torch.linalg.svd_lowrank
                try:
                    U, S, V = torch.svd_lowrank(Wf, q=r_use)
                except Exception:
                    # 回退
                    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
                    V = Vh.T
            else:
                U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
                V = Vh.T
        except RuntimeError as e:
            print(f"[Warn] SVD failed on {k}: {type(e).__name__}: {e}; skipping")
            continue
        # 截断
        U = U[:, :r_use].contiguous()
        S = S[:r_use].contiguous()
        mod_name = _canon_module_from_param(k)
        out[mod_name] = {
            'U': U.cpu(),
            'S': S.cpu(),
            'rank_used': torch.tensor(int(r_use)),
            'param_key': torch.tensor(0),  # 占位符 (保持结构简单)
            'shape': torch.tensor([d_out, d_in])
        }
        kept += 1
        if args.verbose:
            print(f"[SVD] {mod_name}: shape=({d_out},{d_in}) r={r_use}")
    torch.save({'subspaces': out, 'rank': rank, 'model_dir': args.model_dir}, args.save)
    print(f"[Done] Extracted subspaces for {kept}/{total} target linear weights. Saved -> {args.save}")


def parse_args():
    ap = argparse.ArgumentParser(description="GSF Stage-1: Extract truncated SVD subspaces")
    ap.add_argument('--model-dir', type=str, required=True, help='模型目录 (包含权重)')
    ap.add_argument('--save', type=str, required=True, help='输出 .pt 文件')
    ap.add_argument('--rank', type=int, default=64, help='截断 rank (top-r singular vectors)')
    ap.add_argument('--cpu', action='store_true', help='强制使用 CPU (默认自动选择 CUDA)')
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


if __name__ == '__main__':
    extract_subspaces(parse_args())
