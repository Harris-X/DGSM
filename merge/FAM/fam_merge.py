#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlatAlignMerge (FAM) - Step 3: 合并：动态平坦投影融合

基于：
- A(基础) 与 B(捐赠) 模型参数
- FAM 映射文件（由 fam_mapping.py 生成，给出每个模块的神经元对齐 (i_A, j_B)）
- 激活/FAI 统计（由 cache_activation_new.py 生成，用于 d_i 和 Hessian 近似）

对每个可合并的线性层：
- 对映射到的每一对 (i_A, j_B) 行，计算 τ_j = W_B[j,:] - W_A[i,:]
- 投影到 A 的输入方向 d_i，并乘以平坦度权重 (1 - β·H_B[j])
- 合成 W*_i = W_A[i] + λ_proj·τ_proj + λ_ortho·τ_ortho

对 bias：对映射索引按相同平坦权重进行加权差分融合。
未映射的行/元素保持 A。
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, Tuple

import torch
import safetensors
from tqdm import tqdm

from utils import load_weights, need_merge  # 与现有 SAFE 脚本保持一致的工具

EPS = 1e-8


def _canon_module_name(name: str) -> str:
    k = name.replace("language_model.model.", "model.").replace("language_model.", "model.")
    if "layers" in k:
        pos = k.find("layers")
        k = "model." + k[pos:]
    return k


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


def _load_acts(path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    d = torch.load(path, map_location="cpu")
    return { _canon_module_name(k): v for k, v in d.items() }


def _load_mapping(path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    obj = torch.load(path, map_location="cpu")
    mapping = obj.get('mapping', obj)
    # 兼容：规范化模块名
    norm = {}
    for k, v in mapping.items():
        norm[_canon_module_name(k)] = v
    return norm


def _flat_weight(h_val: float, beta: float) -> float:
    # 平坦权重：1 - β·H，裁剪到 [0,1]
    w = 1.0 - beta * float(h_val)
    if w < 0.0:
        return 0.0
    if w > 1.0:
        return 1.0
    return w


def fam_dynamic_flat_merge(args):
    print("\n--- [FAM 阶段三: 动态平坦投影融合] ---")

    # 加载权重
    print("加载 A/B 权重...")
    weights_A = load_weights(args.base_model_path)
    weights_B = load_weights(args.donor_model_path)

    # 构建 B 参数 canonical->orig 键映射（便于按 A 的键找到 B 对应参数）
    b_canon_to_orig: Dict[str, str] = {}
    for k in weights_B.keys():
        ck = _canon_param_key(k)
        if ck not in b_canon_to_orig:
            b_canon_to_orig[ck] = k

    # 加载 FAM 映射
    print("加载 FAM 神经元映射...")
    mapping = _load_mapping(args.mapping_path)

    # 加载激活统计（用于 d_i 与 H_B）
    print("加载激活/FAI 统计...")
    if args.acts_a and osp.exists(args.acts_a):
        acts_A = _load_acts(args.acts_a)
    else:
        # fallback: 从 cache_dir 推断
        a_path = osp.join(args.cache_dir, osp.basename(args.base_model_path).rstrip(os.sep) + "_meta.pt")
        acts_A = _load_acts(a_path)

    if args.acts_b and osp.exists(args.acts_b):
        acts_B = _load_acts(args.acts_b)
    else:
        b_path = osp.join(args.cache_dir, osp.basename(args.donor_model_path).rstrip(os.sep) + "_meta.pt")
        acts_B = _load_acts(b_path)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    lambda_proj = float(args.lambda_proj)
    lambda_ortho = float(args.lambda_ortho)
    lambda_norm = float(getattr(args, 'lambda_norm', 0.0))
    beta = float(args.beta)

    merged = weights_A.copy()

    pbar = tqdm(weights_A.keys(), desc="FAM 合并 (仅 need_merge)")
    for key in pbar:
        if not need_merge(key):
            # 对 LayerNorm / Norm 参数进行简单加权平均（可选）
            if lambda_norm > 0.0 and ('norm' in key.lower() or '.ln' in key.lower() or 'layernorm' in key.lower()):
                a_canon = _canon_param_key(key)
                b_key = b_canon_to_orig.get(a_canon, None)
                if b_key is not None and weights_A[key].shape == weights_B[b_key].shape:
                    W_A = weights_A[key].float()
                    W_B = weights_B[b_key].float()
                    merged[key] = ((1.0 - lambda_norm) * W_A + lambda_norm * W_B).to(weights_A[key].dtype)
            continue

        a_canon = _canon_param_key(key)
        b_key = b_canon_to_orig.get(a_canon, None)
        if b_key is None:
            # B 无对应参数，保持 A
            continue

        W_A = weights_A[key].float()
        W_B = weights_B[b_key].float()
        module_name = _module_from_param_key(key)

        # 仅在有映射时处理
        map_blk = mapping.get(module_name, None)
        if map_blk is None or ('pairs' not in map_blk) or map_blk['pairs'].numel() == 0:
            continue

        pairs = map_blk['pairs'].long()  # [K,2] each is [i_A, j_B]

        if W_A.ndim == 2 and key.endswith('.weight'):
            # 需要 A 的输入方向 d（与 in_features 维一致）
            actsA_blk = acts_A.get(module_name, {})
            d_vec = actsA_blk.get('input', None)
            if d_vec is None:
                # 无方向，跳过该参数
                continue
            d = d_vec.to(device).float()
            if d.dim() != 1 or d.shape[0] != W_A.shape[1]:
                # 尺寸不匹配
                continue

            # B 的平坦度向量（Hessian 近似）
            actsB_blk = acts_B.get(module_name, {})
            H_B = actsB_blk.get('fai_H', None)
            if H_B is not None:
                H_B_np = H_B.view(-1).float().cpu()
            else:
                # 若无 H，使用 0
                H_B_np = torch.zeros(W_B.shape[0], dtype=torch.float32)

            W_A_dev = W_A.to(device)
            W_B_dev = W_B.to(device)

            # 按行更新：i_A <- 使用 j_B
            W_out = W_A_dev.clone()

            d_norm_sq = torch.dot(d, d).clamp_min(EPS)
            for idx in range(pairs.shape[0]):
                i_A = int(pairs[idx, 0].item())
                j_B = int(pairs[idx, 1].item())
                if i_A < 0 or i_A >= W_out.shape[0] or j_B < 0 or j_B >= W_B_dev.shape[0]:
                    continue
                tau = W_B_dev[j_B, :] - W_out[i_A, :]
                proj_scalar = torch.dot(tau, d) / d_norm_sq
                tau_proj = proj_scalar * d
                # 平坦度权重
                h_val = float(H_B_np[j_B]) if j_B < H_B_np.shape[0] else 0.0
                w_flat = _flat_weight(h_val, beta)
                tau_proj = tau_proj * w_flat
                tau_ortho = tau - tau_proj
                W_out[i_A, :] = W_out[i_A, :] + lambda_proj * tau_proj + lambda_ortho * tau_ortho

            merged[key] = W_out.detach().cpu().to(weights_A[key].dtype)

        elif W_A.ndim == 1 and key.endswith('.bias'):
            # bias: 标量更新，使用平坦度权重
            actsB_blk = acts_B.get(module_name, {})
            H_B = actsB_blk.get('fai_H', None)
            if H_B is not None:
                H_B_np = H_B.view(-1).float().cpu()
            else:
                H_B_np = torch.zeros(W_B.shape[0], dtype=torch.float32)

            W_A_dev = W_A.to(device)
            W_B_dev = W_B.to(device)
            W_out = W_A_dev.clone()

            for idx in range(pairs.shape[0]):
                i_A = int(pairs[idx, 0].item())
                j_B = int(pairs[idx, 1].item())
                if i_A < 0 or i_A >= W_out.shape[0] or j_B < 0 or j_B >= W_B_dev.shape[0]:
                    continue
                tau = W_B_dev[j_B] - W_out[i_A]
                h_val = float(H_B_np[j_B]) if j_B < H_B_np.shape[0] else 0.0
                w_flat = _flat_weight(h_val, beta)
                # 这里将“投影”简化为全部差分，并应用平坦权重；无正交分解（标量无方向差异）
                W_out[i_A] = W_out[i_A] + lambda_proj * (tau * w_flat)

            merged[key] = W_out.detach().cpu().to(weights_A[key].dtype)

        else:
            # 其他参数形状暂不处理
            continue

    _save_model(args, merged)


def _save_model(args, merged_weights):
    """保存模型权重（兼容 safetensors/bin 分片）。"""
    print("\n正在保存合并后的模型...")
    output_dir = osp.basename(args.base_model_path.rstrip(os.sep))
    output_dir = osp.join(args.output_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sft_index = os.path.join(args.base_model_path, "model.safetensors.index.json")
    bin_index = os.path.join(args.base_model_path, "pytorch_model.bin.index.json")

    def copy_side_files():
        for filename in os.listdir(args.base_model_path):
            if filename.endswith((".json", ".model", ".py", ".md")):
                src = os.path.join(args.base_model_path, filename)
                dst = os.path.join(output_dir, filename)
                if not os.path.exists(dst):
                    try:
                        import shutil
                        shutil.copy(src, dst)
                    except Exception:
                        pass

    if os.path.exists(sft_index):
        with open(sft_index, "r") as f:
            index_map = json.load(f)["weight_map"]
        sharded_weights = defaultdict(dict)
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        for filename, weights_dict in sharded_weights.items():
            safetensors.torch.save_file(weights_dict, os.path.join(output_dir, filename))
        try:
            import shutil
            shutil.copy(sft_index, os.path.join(output_dir, os.path.basename(sft_index)))
        except Exception:
            pass
        copy_side_files()
        print(f"模型成功合并并保存至: {output_dir} (safetensors 分片)")
        return

    if os.path.exists(bin_index):
        with open(bin_index, "r") as f:
            index_map = json.load(f)["weight_map"]
        sharded_weights = defaultdict(dict)
        for key, value in merged_weights.items():
            if key in index_map:
                sharded_weights[index_map[key]][key] = value
        for filename, weights_dict in sharded_weights.items():
            torch.save(weights_dict, os.path.join(output_dir, filename))
        try:
            import shutil
            shutil.copy(bin_index, os.path.join(output_dir, os.path.basename(bin_index)))
        except Exception:
            pass
        copy_side_files()
        print(f"模型成功合并并保存至: {output_dir} (.bin 分片)")
        return

    # Fallback: single-file save
    sft_single = os.path.join(args.base_model_path, "model.safetensors")
    bin_single = os.path.join(args.base_model_path, "pytorch_model.bin")
    if os.path.exists(sft_single):
        out_path = os.path.join(output_dir, os.path.basename(sft_single))
        safetensors.torch.save_file(merged_weights, out_path)
        copy_side_files()
        print(f"模型成功合并并保存至: {out_path} (单一 safetensors)")
        return
    if os.path.exists(bin_single):
        out_path = os.path.join(output_dir, os.path.basename(bin_single))
        torch.save(merged_weights, out_path)
        copy_side_files()
        print(f"模型成功合并并保存至: {out_path} (单一 .bin)")
        return

    out_path = os.path.join(output_dir, "model.safetensors")
    safetensors.torch.save_file(merged_weights, out_path)
    copy_side_files()
    print(f"模型成功合并并保存至: {out_path} (默认 safetensors)")


def main():
    parser = argparse.ArgumentParser(description="FAM Step3: 动态平坦投影融合")
    parser.add_argument('--base_model_path', type=str, required=True, help='基础模型A路径')
    parser.add_argument('--donor_model_path', type=str, required=True, help='捐赠模型B路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')

    parser.add_argument('--mapping_path', type=str, required=True, help='FAM 映射文件（fam_mapping.py 输出的 .pt）')
    parser.add_argument('--acts_a', type=str, default=None, help='A 模型激活/FAI缓存（.pt），未提供则按 cache_dir 推断')
    parser.add_argument('--acts_b', type=str, default=None, help='B 模型激活/FAI缓存（.pt），未提供则按 cache_dir 推断')
    parser.add_argument('--cache_dir', type=str, default='activations', help='激活/FAI缓存目录（用于推断路径）')

    parser.add_argument('--lambda_proj', type=float, default=1.0, help='投影分量系数')
    parser.add_argument('--lambda_ortho', type=float, default=0.8, help='正交分量系数')
    parser.add_argument('--beta', type=float, default=0.5, help='平坦惩罚系数 β')
    parser.add_argument('--lambda_norm', type=float, default=0.0, help='LayerNorm 等归一化参数的加权平均系数')
    parser.add_argument('--device', type=str, default=None, help='设备，如 cuda:0 或 cpu；默认自动选择')

    args = parser.parse_args()
    fam_dynamic_flat_merge(args)


if __name__ == '__main__':
    main()
