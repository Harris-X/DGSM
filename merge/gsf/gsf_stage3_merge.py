#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSF-TEFM Stage-3: Subspace-Gromov Fusion Merge

输入：
  --base-model  模型A目录
  --donor-model 模型B目录
  --stage2      Stage-2 输出 (包含每个模块的 pi, psi_A, psi_B, S_A, S_B)

过程：
  1. 读取权重 W_A, W_B。
  2. 对每个可合并线性层 (need_merge 且 .weight 2D)：
       - 若存在 Stage-2 对齐 (pi) 则：
           * 对 W_A, W_B 做截断 SVD (top-rA, top-rB) (或复用 Stage-1 结果；此处为自包含再算一次)
           * 取投影差分 τ = W_B - W_A
           * 计算 A 子空间投影: τ_proj = U_A (U_A^T τ)
           * 正交残差: τ_ortho = τ - τ_proj
           * 计算融合权重 λ：λ = sigmoid( γ * (1 - cost_norm) ) 其中 cost_norm = min(1, gwd_cost / cost_scale)
             - cost_scale 为命令行参数 (默认 1.0)
             - γ (gamma) 控制陡峭程度 (默认 4.0)
           * 平衡: W*_A = W_A + λ * τ_proj + λ_ortho * τ_ortho
             其中 λ_ortho = λ * ortho_scale (小于 1 防止噪声)
       - 否则回退简单加权: W*_A = (1-alpha)*W_A + alpha*W_B
     Bias 类似 (仅整体差分加权)。
  3. 保存新模型到 --output-dir/<base_model_basename>/gsf_merged/

理由：
  - 使用 GWD cost 作为子空间距离；较小 cost 表示更可信的对齐 -> 更大 λ。
  - 投影部分注入“共享功能”差分，正交部分衰减注入以避免噪声。

扩展 (2025-09-27):
    - 支持可选 Stage-2b 行级 / 组级 λ (row_lambda) 对 τ_proj 进行逐行缩放:
             若提供 --stage2b 文件且包含对应模块 row_lambda:
                    τ_proj_scaled[i,:] = row_lambda[i] * τ_proj[i,:]
                    最终: W_new = W_A + τ_proj_scaled * λ + τ_ortho * (λ * ortho_scale)
        作用: 更细粒度注入保留高贡献神经元，抑制低贡献噪声。

输出：
    - safetensors / bin 与原格式一致的合并模型权重
    - merge_meta_gsf.json 记录超参数与统计 (新增 row_lambda_used)
"""
from __future__ import annotations
import argparse
import json
import math
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, Optional

import torch
import safetensors
import safetensors.torch
from tqdm import tqdm

from utils import load_weights, need_merge  # type: ignore

EPS = 1e-8


def _canon_param_key(param_key: str) -> str:
    k = param_key.replace("language_model.model.", "model.").replace("language_model.", "model.")
    if "layers" in k:
        pos = k.find("layers")
        k = "model." + k[pos:]
    return k


def _module_from_param_key(param_key: str) -> str:
    k = _canon_param_key(param_key)
    parts = k.split('.')
    if len(parts) >= 2:
        parts = parts[:-1]
    return '.'.join(parts)


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _svd_trunc(W: torch.Tensor, r: int):
    Wf = W.float()
    r_use = min(r, Wf.shape[0], Wf.shape[1])
    if r_use <= 0:
        return None
    try:
        U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
        return U[:, :r_use].contiguous(), S[:r_use].contiguous()
    except Exception:
        return None


def _save_model(args, merged_weights: Dict[str, torch.Tensor]):
    base_dir = osp.basename(args.base_model.rstrip(os.sep))
    out_root = osp.join(args.output_dir, base_dir, 'gsf_merged')
    os.makedirs(out_root, exist_ok=True)
    # 处理分片
    sft_index = osp.join(args.base_model, 'model.safetensors.index.json')
    bin_index = osp.join(args.base_model, 'pytorch_model.bin.index.json')

    def copy_side():
        for fn in os.listdir(args.base_model):
            if fn.endswith(('.json', '.model', '.py', '.md')):
                try:
                    src = osp.join(args.base_model, fn)
                    dst = osp.join(out_root, fn)
                    if not osp.exists(dst):
                        import shutil
                        shutil.copy(src, dst)
                except Exception:
                    pass

    if osp.exists(sft_index):
        with open(sft_index, 'r') as f:
            index = json.load(f)["weight_map"]
        shards = defaultdict(dict)
        for k, v in merged_weights.items():
            if k in index:
                shards[index[k]][k] = v
        for shard_name, shard_dict in shards.items():
            safetensors.torch.save_file(shard_dict, osp.join(out_root, shard_name))
        copy_side()
        print(f"[Save] Sharded safetensors -> {out_root}")
        return
    if osp.exists(bin_index):
        with open(bin_index, 'r') as f:
            index = json.load(f)["weight_map"]
        shards = defaultdict(dict)
        for k, v in merged_weights.items():
            if k in index:
                shards[index[k]][k] = v
        for shard_name, shard_dict in shards.items():
            torch.save(shard_dict, osp.join(out_root, shard_name))
        copy_side(); print(f"[Save] Sharded .bin -> {out_root}")
        return
    # 单文件 fallback
    sft_single = osp.join(args.base_model, 'model.safetensors')
    bin_single = osp.join(args.base_model, 'pytorch_model.bin')
    if osp.exists(sft_single):
        safetensors.torch.save_file(merged_weights, osp.join(out_root, 'model.safetensors'))
        copy_side(); print(f"[Save] Single safetensors -> {out_root}")
        return
    if osp.exists(bin_single):
        torch.save(merged_weights, osp.join(out_root, 'pytorch_model.bin'))
        copy_side(); print(f"[Save] Single .bin -> {out_root}")
        return
    safetensors.torch.save_file(merged_weights, osp.join(out_root, 'model.safetensors'))
    copy_side(); print(f"[Save] Default safetensors -> {out_root}")


def gsf_merge(args: argparse.Namespace):
    print("\n--- [GSF Stage-3: Subspace-Gromov Fusion Merge] ---")
    weights_A = load_weights(args.base_model)
    weights_B = load_weights(args.donor_model)
    stage2 = torch.load(args.stage2, map_location='cpu')
    modules_info = stage2.get('modules', stage2)
    stage2b = None
    if args.stage2b is not None and os.path.exists(args.stage2b):
        try:
            obj2b = torch.load(args.stage2b, map_location='cpu')
            stage2b = obj2b.get('modules', obj2b)
            print(f"[Info] Loaded Stage-2b group localization: {args.stage2b}")
        except Exception as e:
            print(f"[Warn] Failed loading stage2b ({e}); ignore row-level λ")

    merged = weights_A.copy()
    stat_layers = 0
    stat_used = 0
    stat_row_used = 0

    for k in tqdm(list(weights_A.keys()), desc='GSF Merge'):
        if not need_merge(k):
            continue
        if k.endswith('.weight') and weights_A[k].ndim == 2:
            mod_name = _module_from_param_key(k)
            blk = modules_info.get(mod_name)
            W_A = weights_A[k].float()
            W_B = weights_B.get(k, None)
            if W_B is None:
                continue
            W_B = W_B.float()
            stat_layers += 1
            if blk is not None:
                gwd_cost = float(blk['gwd_cost'])
                # 规范化 cost
                cost_scale = max(EPS, float(args.cost_scale))
                cost_norm = min(1.0, gwd_cost / cost_scale)
                # λ 根据 cost 反比调节 (cost 越低 λ 越大)
                gamma = float(args.gamma)
                lam = _sigmoid(gamma * (1.0 - cost_norm))
                ortho_scale = float(args.ortho_scale)
                # 仅用 A 的子空间 (重新计算 SVD 以获得投影矩阵)
                rA = int(blk['rank_A'])
                svdA = _svd_trunc(W_A, rA)
                if svdA is None:
                    continue
                U_A, S_A = svdA
                # 差分
                tau = W_B - W_A
                tau_proj = U_A @ (U_A.T @ tau)
                tau_ortho = tau - tau_proj
                # 行级 λ (可选)
                if stage2b is not None:
                    blk2b = stage2b.get(mod_name)
                    if blk2b is not None and 'row_lambda' in blk2b:
                        row_lambda = blk2b['row_lambda'].float().to(tau_proj.device)
                        if row_lambda.shape[0] == tau_proj.shape[0]:
                            tau_proj = row_lambda.unsqueeze(1) * tau_proj
                            stat_row_used += 1
                W_new = W_A + lam * tau_proj + (lam * ortho_scale) * tau_ortho
                merged[k] = W_new.to(weights_A[k].dtype)
                stat_used += 1
            else:
                # Fallback simple blend
                alpha = float(args.fallback_alpha)
                merged[k] = (1 - alpha) * W_A + alpha * W_B
        elif k.endswith('.bias') and weights_A[k].ndim == 1:
            # bias 简单 blend (或与 gwd_cost 调节)
            W_A = weights_A[k].float()
            W_B = weights_B.get(k, None)
            if W_B is None:
                continue
            W_B = W_B.float()
            alpha = float(args.bias_alpha)
            merged[k] = (1 - alpha) * W_A + alpha * W_B

    _save_model(args, merged)
    meta = dict(
        base_model=args.base_model,
        donor_model=args.donor_model,
        stage2=args.stage2,
        stage2b=args.stage2b,
        cost_scale=float(args.cost_scale),
        gamma=float(args.gamma),
        ortho_scale=float(args.ortho_scale),
        fallback_alpha=float(args.fallback_alpha),
        bias_alpha=float(args.bias_alpha),
        stat_layers=stat_layers,
        stat_used=stat_used,
        stat_row_lambda_used=stat_row_used,
    )
    base_dir = osp.basename(args.base_model.rstrip(os.sep))
    out_root = osp.join(args.output_dir, base_dir, 'gsf_merged')
    os.makedirs(out_root, exist_ok=True)
    with open(osp.join(out_root, 'merge_meta_gsf.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[Done] GSF merge complete: layers={stat_layers}, used_gwd={stat_used}")


def parse_args():
    ap = argparse.ArgumentParser(description="GSF Stage-3: Subspace-Gromov Fusion Merge")
    ap.add_argument('--base-model', type=str, required=True)
    ap.add_argument('--donor-model', type=str, required=True)
    ap.add_argument('--stage2', type=str, required=True)
    ap.add_argument('--stage2b', type=str, default=None, help='可选：Stage-2b 行/组 λ 文件')
    ap.add_argument('--output-dir', type=str, required=True)
    # λ & cost 调节
    ap.add_argument('--cost-scale', type=float, default=1.0, help='GWD cost 归一化尺度 (cost/cost_scale)')
    ap.add_argument('--gamma', type=float, default=4.0, help='λ sigmoid 陡峭度')
    ap.add_argument('--ortho-scale', type=float, default=0.5, help='正交残差缩放 (乘以 λ)')
    ap.add_argument('--fallback-alpha', type=float, default=0.5, help='缺少对齐信息时的权重 blend 系数')
    ap.add_argument('--bias-alpha', type=float, default=0.5, help='bias 融合 alpha')
    return ap.parse_args()


if __name__ == '__main__':
    gsf_merge(parse_args())
