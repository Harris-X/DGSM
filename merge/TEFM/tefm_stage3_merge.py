#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEFM Stage-3: 轨迹 / 平坦 / 互信息 (TFI) 引导的干扰抑制融合 (Ensemble → Neuron Merge)

目标：在已有
  1) 模型 A (base) 与 模型 B (donor) 权重
  2) Stage-2 输出的 ensemble 映射文件 (tefm_stage2_ensemble.py 生成)
  3) Stage-1 LAPE sidecar (提供 phi_a / phi_g / phi_L)
  4) 可选 averaged activations + FAI (提供 input 方向、Hessian 近似 fai_H)
基础上，执行更细粒度的神经元级参数融合：

核心思想：
  - 先从 ensemble 对齐 (E^A_k ↔ E^B_p) 反推神经元对齐方案。
  - 对每个被对齐的 (i_A, j_B) 行权重，构造差分 τ = W_B[j] - W_A[i]。
  - 分解 τ 为：平行于 A 的输入方向 d_i 的部分 (投影) 与 正交部分 (残差)。
  - 引入多重权重：
        w_flat      : 平坦度惩罚   = exp(-beta_h * |h_A(i) - h_B(j)|) 或使用 (1 - beta_flat * H_B[j]) 截断
        w_tfi       : 重要性放缩   = TFI_B[j] / (mean_TFI_B + eps)
        w_conflict  : 梯度冲突门控 = sigmoid(alpha_conflict * sign(phi_g_A[i] * phi_g_B[j])) (冲突时降低)
        w_mi        : MI 放缩 (可选) = mi_AB(j) / (mean_mi_B + eps) 若 Stage-2 提供
    最终权重 w = w_flat * w_tfi * w_conflict * w_mi
  - 融合：W_A[i] <- W_A[i] + λ_proj * (w * τ_proj) + λ_ortho * (w * τ_ortho)
  - bias 类似但无方向分解。

神经元匹配 (从 ensemble 映射到单元)：提供多种模式：
  --pair-mode cartesian: 对每个 (kA,kB) 对的所有 IA × JB 笛卡尔积 (可能较大)
  --pair-mode round_robin (默认): 将 IA 与 JB 按索引轮转配对，保持数量=min(|IA|,|JB|)
  --pair-mode size-min: 若 |IA| != |JB|，随机下采样较大的集合，再一一配对
  --pair-mode centroid: 不生成神经元对，用 ensemble 均值差更新所有 IA (广播) （快速近似）

输入文件：
  * --stage2 path/to/stage2_output.pt  (包含 modules[name].ensembles_A / ensembles_B / pairs / tfi_A / tfi_B / h_mean_A / h_mean_B ...)
  * --base-model / --donor-model  (模型目录，支持 safetensors / bin 分片，与 fam_merge 兼容的 load_weights 工具)
  * --base-lape / --donor-lape    (LAPE sidecar .pt; 若为 None 自动根据 averaged 文件推断 *_lape.pt)
  * --acts-base / --acts-donor    (averaged activations，提供 input 向量与 fai_H；可缺省)

输出：
  * 新模型 (默认写入 --output-dir/<base_model_basename>/ ) 包含融合后的权重
  * 伴随保存一个 merge_meta.json 记录参数与统计

假设：
  - utils.load_weights, utils.need_merge 已存在（与 fam_merge.py 相同）
  - LAPE sidecar 中 phi_a/phi_g/phi_L 形状一致 (H,)
  - Stage-2 文件结构与 tefm_stage2_ensemble.py 描述一致

注意：
  - 若未找到方向 d_i (缺少 averaged_activations[input])，仅使用全差分 (视为纯正交更新)；会打印告警。
  - 所有张量在处理时转换为 float32，再回写为原 dtype。

后续可扩展：
  - 自适应 λ_proj / λ_ortho 调度 (基于 w 分布)
  - 任务特定屏蔽 (按 TFI 阈值)
  - 多模型 (>2) 级联 / 混合
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import safetensors
from tqdm import tqdm

# 依赖已有工具 (复用 fam_merge.py 假设的接口)
from utils import load_weights, need_merge  # type: ignore

# 改进 (2025-09-27):
# - 增强命名规范统一: 兼容 llama_model./model.model./base_model.model.
# - 添加确定性选项、冲突幅值比、更新幅度约束。

EPS = 1e-8

# ---------------- Utility: canonical naming (与 Stage-2 / fam_merge 保持一致) ----------------

def _canon_module_name(name: str) -> str:
    k = name.replace("language_model.model.", "model.").replace("language_model.", "model.")
    k = k.replace("llama_model.", "model.").replace("model.model.", "model.").replace("base_model.model.", "model.")
    if "layers" in k:
        pos = k.find("layers")
        k = "model." + k[pos:]
    return k

def _canon_param_key(param_key: str) -> str:
    k = param_key.replace("language_model.model.", "model.").replace("language_model.", "model.")
    k = k.replace("llama_model.", "model.").replace("model.model.", "model.").replace("base_model.model.", "model.")
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

# ---------------- Loading helpers ----------------

def _load_pt(path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    d = torch.load(path, map_location='cpu')
    return { _canon_module_name(k): v for k, v in d.items() }

def _load_stage2(path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    obj = torch.load(path, map_location='cpu')
    return obj.get('modules', obj)

def _infer_sidecar(base: str) -> str:
    b, e = osp.splitext(base)
    cand = b + "_lape" + e
    return cand if osp.exists(cand) else ""

# ---------------- Pair expansion from ensembles ----------------

def _expand_pairs(labels_A: torch.Tensor,
                  labels_B: torch.Tensor,
                  ens_pairs: torch.Tensor,
                  mode: str = 'round_robin',
                  rng: random.Random | None = None,
                  deterministic: bool = False) -> List[Tuple[int, int]]:
    """ 将 ensemble 对 (kA,kB) 展开成神经元对列表。
    labels_*: LongTensor[H_*]  每个神经元所属簇
    ens_pairs: LongTensor[M,2] 匹配的 (kA,kB)
    返回: list[(i_A, j_B)]
    """
    if rng is None:
        rng = random.Random(42)
    out: List[Tuple[int,int]] = []
    for (kA, kB) in ens_pairs.tolist():
        IA = (labels_A == kA).nonzero(as_tuple=False).view(-1).tolist()
        JB = (labels_B == kB).nonzero(as_tuple=False).view(-1).tolist()
        if not IA or not JB:
            continue
        if mode == 'cartesian':
            for i in IA:
                for j in JB:
                    out.append((i, j))
        elif mode == 'size-min':
            n = min(len(IA), len(JB))
            if deterministic:
                IA_sel = IA[:n]
                JB_sel = JB[:n]
            else:
                IA_sel = IA if len(IA) == n else rng.sample(IA, n)
                JB_sel = JB if len(JB) == n else rng.sample(JB, n)
            for idx in range(n):
                out.append((IA_sel[idx], JB_sel[idx]))
        elif mode == 'centroid':
            # 使用 "伪配对": 与 IA 中每个 i 形成一一映射到 JB[ idx % len(JB) ]，但后续会对 ensemble 均值进行更新
            # 此模式实际在后续融合逻辑中检测并走 ensemble 广播分支。
            for i in IA:
                out.append((i, JB[0]))  # 记录一个代表 j，仅作占位
        else:  # round_robin
            n = min(len(IA), len(JB))
            for idx in range(n):
                out.append((IA[idx % len(IA)], JB[idx % len(JB)]))
    return out

# ---------------- Merge core ----------------

def _flat_weight(h_val: float, beta_flat: float) -> float:
    # 线性裁剪方式 (与 fam_merge 类似)；可被 exp 曲率版本替代
    w = 1.0 - beta_flat * float(h_val)
    if w < 0.0: return 0.0
    if w > 1.0: return 1.0
    return w

def _curv_penalty(hA: float, hB: float, beta_h: float) -> float:
    return math.exp(-beta_h * abs(hA - hB))

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# ---------------- Main merge procedure ----------------

def tefm_stage3_merge(args):
    print("\n--- [TEFM Stage-3: 轨迹平坦干扰抑制融合] ---")

    # 1. 载入权重
    print("加载 A/B 权重...")
    weights_A = load_weights(args.base_model)
    weights_B = load_weights(args.donor_model)

    # donor 原始键索引 (方便回溯)
    b_canon_to_orig = {}
    for k in weights_B.keys():
        ck = _canon_param_key(k)
        if ck not in b_canon_to_orig:
            b_canon_to_orig[ck] = k

    # 2. 载入 Stage-2 ensemble 映射
    print("加载 Stage-2 映射文件...")
    stage2 = _load_stage2(args.stage2)

    # 3. 载入 averaged activations / FAI (可选) 用于 input 方向 & H_donor
    acts_A = _load_pt(args.acts_base) if args.acts_base and osp.exists(args.acts_base) else {}
    acts_B = _load_pt(args.acts_donor) if args.acts_donor and osp.exists(args.acts_donor) else {}

    # 4. 载入 LAPE sidecar (phi vectors)
    base_lape = args.base_lape or _infer_sidecar(args.acts_base or "")
    donor_lape = args.donor_lape or _infer_sidecar(args.acts_donor or "")
    lape_A = _load_pt(base_lape) if base_lape and osp.exists(base_lape) else {}
    lape_B = _load_pt(donor_lape) if donor_lape and osp.exists(donor_lape) else {}

    # 设备
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 参数
    beta_flat = float(args.beta_flat)
    beta_h = float(args.beta_h)
    lambda_proj = float(args.lambda_proj)
    lambda_ortho = float(args.lambda_ortho)
    alpha_conflict = float(args.alpha_conflict)
    use_curv_exp = bool(args.use_curv_exp)

    pair_mode = args.pair_mode

    merged = weights_A.copy()

    # 统计
    stat_pairs = 0
    stat_modules = 0

    # 遍历参数 (与 fam_merge 一致只处理 need_merge 的线性/偏置)
    pbar = tqdm(weights_A.keys(), desc="TEFM Stage-3 合并")
    for key in pbar:
        if not need_merge(key):
            # 归一化层可选平均
            if args.lambda_norm > 0.0 and ('norm' in key.lower() or '.ln' in key.lower() or 'layernorm' in key.lower()):
                a_ck = _canon_param_key(key)
                b_key = b_canon_to_orig.get(a_ck, None)
                if b_key is not None and weights_A[key].shape == weights_B[b_key].shape:
                    merged[key] = ((1.0 - args.lambda_norm) * weights_A[key].float() + args.lambda_norm * weights_B[b_key].float()).to(weights_A[key].dtype)
            continue

        a_ck = _canon_param_key(key)
        b_key = b_canon_to_orig.get(a_ck, None)
        if b_key is None:
            continue

        W_A = weights_A[key].float()
        W_B = weights_B[b_key].float()
        module_name = _module_from_param_key(key)
        mod_canon = _canon_module_name(module_name)

        stage2_blk = stage2.get(mod_canon, None)
        if stage2_blk is None:
            continue

        # ensemble labels + pairs
        labels_A = stage2_blk.get('ensembles_A', None)
        labels_B = stage2_blk.get('ensembles_B', None)
        ens_pairs = stage2_blk.get('pairs', None)
        if labels_A is None or labels_B is None or ens_pairs is None or ens_pairs.numel() == 0:
            continue

        # per-neuron tfi if present
        tfi_A = stage2_blk.get('tfi_A', None)
        tfi_B = stage2_blk.get('tfi_B', None)
        if tfi_A is None and mod_canon in lape_A and 'phi_a' in lape_A[mod_canon]:
            # 重建一个简易 TFI (当 Stage-2 未写入时)
            blkA = lape_A[mod_canon]
            blkB = lape_B.get(mod_canon, {})
            if 'phi_a' in blkA and 'phi_g' in blkA and 'phi_L' in blkA:
                tfi_A = blkA['phi_g'].abs() * blkA['phi_a'].abs() / (blkA['phi_g'].pow(2) + blkA['phi_L'] + 1e-6)
            if 'phi_a' in blkB and 'phi_g' in blkB and 'phi_L' in blkB:
                tfi_B = blkB['phi_g'].abs() * blkB['phi_a'].abs() / (blkB['phi_g'].pow(2) + blkB['phi_L'] + 1e-6)
        # curvature (H) from FAI if available
        H_A = acts_A.get(mod_canon, {}).get('fai_H', None)
        H_B = acts_B.get(mod_canon, {}).get('fai_H', None)

        # 方向向量 d (输入激活平均) 来自 A
        d_vec = acts_A.get(mod_canon, {}).get('input', None)
        if d_vec is not None:
            d = d_vec.to(device).float()
            if d.dim() != 1 or (W_A.ndim == 2 and d.shape[0] != W_A.shape[1]):
                d = None  # 尺寸不匹配
        else:
            d = None

        # LAPE phi_g 用于冲突检测 (符号)
        phi_g_A = lape_A.get(mod_canon, {}).get('phi_g', None)
        phi_g_B = lape_B.get(mod_canon, {}).get('phi_g', None)

        # 展开神经元对
        rng = random.Random(1234)
        deterministic = bool(getattr(args, 'deterministic', False)) or bool(getattr(args, 'no_random_pairs', False))
        neuron_pairs = _expand_pairs(labels_A, labels_B, ens_pairs, mode=pair_mode, rng=rng, deterministic=deterministic)
        if not neuron_pairs:
            continue

        # 约束参数
        max_row_ratio = float(getattr(args, 'max_row_change_ratio', 0.25))
        module_max_ratio = float(getattr(args, 'module_max_change_ratio', 0.35))
        conflict_mag_gamma = float(getattr(args, 'conflict_mag_gamma', 0.5))

        if W_A.ndim == 2 and key.endswith('.weight'):
            W_A_dev = W_A.to(device)
            W_B_dev = W_B.to(device)
            W_orig = W_A_dev.clone()
            W_out = W_A_dev.clone()

            # 预计算归一化因子
            tfi_B_arr = tfi_B.float().cpu() if isinstance(tfi_B, torch.Tensor) else None
            mean_tfi_B = float(tfi_B_arr.mean()) if tfi_B_arr is not None and tfi_B_arr.numel() > 0 else 1.0
            phi_g_A_arr = phi_g_A.float().cpu() if isinstance(phi_g_A, torch.Tensor) else None
            phi_g_B_arr = phi_g_B.float().cpu() if isinstance(phi_g_B, torch.Tensor) else None

            # 逐对融合
            d_norm_sq = torch.dot(d, d).clamp_min(EPS) if d is not None else None
            for (i_A, j_B) in neuron_pairs:
                if i_A < 0 or i_A >= W_out.shape[0] or j_B < 0 or j_B >= W_B_dev.shape[0]:
                    continue
                # 基于原始行差分，避免顺序依赖
                tau_full = W_B_dev[j_B, :] - W_orig[i_A, :]
                if d is not None and d_norm_sq is not None:
                    proj_scalar = torch.dot(tau_full, d) / d_norm_sq
                    tau_proj = proj_scalar * d
                    tau_ortho = tau_full - tau_proj
                else:
                    tau_proj = torch.zeros_like(tau_full)
                    tau_ortho = tau_full

                # 权重计算 -------------------------------------------------
                # 平坦度：两种策略 (取其一) -> flat_weight (线性) + curvature penalty (指数)
                h_val_B = float(H_B[j_B].item()) if (H_B is not None and j_B < H_B.shape[0]) else 0.0
                w_flat_lin = _flat_weight(h_val_B, beta_flat) if beta_flat > 0 else 1.0
                if use_curv_exp and H_A is not None and H_B is not None and i_A < H_A.shape[0] and j_B < H_B.shape[0]:
                    w_curv = _curv_penalty(float(H_A[i_A].item()), float(H_B[j_B].item()), beta_h)
                else:
                    w_curv = 1.0
                # TFI 权重
                if tfi_B_arr is not None and j_B < tfi_B_arr.shape[0] and mean_tfi_B > 0:
                    w_tfi = float(tfi_B_arr[j_B].item()) / mean_tfi_B
                else:
                    w_tfi = 1.0
                # 冲突 (梯度方向)：符号相同 -> 增强  / 相反 -> 减弱
                w_conflict = 1.0
                if (phi_g_A_arr is not None and phi_g_B_arr is not None
                    and i_A < phi_g_A_arr.shape[0] and j_B < phi_g_B_arr.shape[0]):
                    gA = phi_g_A_arr[i_A].item(); gB = phi_g_B_arr[j_B].item()
                    sign_prod = gA * gB
                    w_conflict = _sigmoid(alpha_conflict * (1.0 if sign_prod >= 0 else -1.0))
                    # 幅值匹配增强/抑制 (接近 -> 1, 相差大 -> <1)
                    if abs(gA) > 0 and abs(gB) > 0:
                        mag_ratio = min(abs(gA), abs(gB)) / (max(abs(gA), abs(gB)) + EPS)
                        w_conflict = w_conflict * (mag_ratio ** conflict_mag_gamma)
                # 总权重
                w = w_flat_lin * w_curv * w_tfi * w_conflict
                if w <= 0:
                    continue
                proposed_delta = lambda_proj * (w * tau_proj) + lambda_ortho * (w * tau_ortho)
                current_delta = W_out[i_A, :] - W_orig[i_A, :]
                new_delta = current_delta + proposed_delta
                base_norm = W_orig[i_A, :].norm().item() + EPS
                max_allowed = max_row_ratio * base_norm
                if max_allowed > 0:
                    delta_norm = new_delta.norm().item()
                    if delta_norm > max_allowed:
                        new_delta = new_delta * (max_allowed / delta_norm)
                W_out[i_A, :] = W_orig[i_A, :] + new_delta
            merged[key] = W_out.detach().cpu().to(weights_A[key].dtype)
            stat_pairs += len(neuron_pairs)
            stat_modules += 1

            # 模块级整体约束
            mod_delta = W_out - W_orig
            mod_ratio = mod_delta.norm().item() / (W_orig.norm().item() + EPS)
            if module_max_ratio > 0 and mod_ratio > module_max_ratio:
                scale = module_max_ratio / mod_ratio
                W_out = W_orig + mod_delta * scale
                merged[key] = W_out.detach().cpu().to(weights_A[key].dtype)

        elif W_A.ndim == 1 and key.endswith('.bias'):
            W_A_dev = W_A.to(device)
            W_B_dev = W_B.to(device)
            W_orig = W_A_dev.clone()
            W_out = W_A_dev.clone()
            tfi_B_arr = tfi_B.float().cpu() if isinstance(tfi_B, torch.Tensor) else None
            mean_tfi_B = float(tfi_B_arr.mean()) if tfi_B_arr is not None and tfi_B_arr.numel() > 0 else 1.0
            phi_g_A_arr = phi_g_A.float().cpu() if isinstance(phi_g_A, torch.Tensor) else None
            phi_g_B_arr = phi_g_B.float().cpu() if isinstance(phi_g_B, torch.Tensor) else None
            for (i_A, j_B) in neuron_pairs:
                if i_A < 0 or i_A >= W_out.shape[0] or j_B < 0 or j_B >= W_B_dev.shape[0]:
                    continue
                tau_full = W_B_dev[j_B] - W_orig[i_A]
                h_val_B = float(H_B[j_B].item()) if (H_B is not None and j_B < H_B.shape[0]) else 0.0
                w_flat_lin = _flat_weight(h_val_B, beta_flat) if beta_flat > 0 else 1.0
                if use_curv_exp and H_A is not None and H_B is not None and i_A < H_A.shape[0] and j_B < H_B.shape[0]:
                    w_curv = _curv_penalty(float(H_A[i_A].item()), float(H_B[j_B].item()), beta_h)
                else:
                    w_curv = 1.0
                if tfi_B_arr is not None and j_B < tfi_B_arr.shape[0] and mean_tfi_B > 0:
                    w_tfi = float(tfi_B_arr[j_B].item()) / mean_tfi_B
                else:
                    w_tfi = 1.0
                w_conflict = 1.0
                if (phi_g_A_arr is not None and phi_g_B_arr is not None
                    and i_A < phi_g_A_arr.shape[0] and j_B < phi_g_B_arr.shape[0]):
                    gA = phi_g_A_arr[i_A].item(); gB = phi_g_B_arr[j_B].item()
                    sign_prod = gA * gB
                    w_conflict = _sigmoid(alpha_conflict * (1.0 if sign_prod >= 0 else -1.0))
                    if abs(gA) > 0 and abs(gB) > 0:
                        mag_ratio = min(abs(gA), abs(gB)) / (max(abs(gA), abs(gB)) + EPS)
                        w_conflict = w_conflict * (mag_ratio ** conflict_mag_gamma)
                w = w_flat_lin * w_curv * w_tfi * w_conflict
                if w <= 0:
                    continue
                proposed_delta = lambda_proj * w * tau_full
                current_delta = W_out[i_A] - W_orig[i_A]
                new_delta = current_delta + proposed_delta
                base_norm = abs(W_orig[i_A].item()) + EPS
                max_allowed = max_row_ratio * base_norm
                if max_allowed > 0 and abs(new_delta.item()) > max_allowed:
                    new_delta = new_delta.sign() * max_allowed
                W_out[i_A] = W_orig[i_A] + new_delta
            merged[key] = W_out.detach().cpu().to(weights_A[key].dtype)
            stat_pairs += len(neuron_pairs)
            stat_modules += 1
            # 模块级 (bias 向量) 约束
            delta_vec = W_out - W_orig
            ratio = delta_vec.norm().item() / (W_orig.norm().item() + EPS)
            if module_max_ratio > 0 and ratio > module_max_ratio:
                scale = module_max_ratio / ratio
                W_out = W_orig + delta_vec * scale
                merged[key] = W_out.detach().cpu().to(weights_A[key].dtype)
        else:
            # 未处理结构
            continue

    # 5. 保存
    _save_model(args, merged)

    meta = dict(
        base_model=args.base_model,
        donor_model=args.donor_model,
        stage2=args.stage2,
        acts_base=args.acts_base,
        acts_donor=args.acts_donor,
        base_lape=base_lape,
        donor_lape=donor_lape,
        beta_flat=beta_flat,
        beta_h=beta_h,
        lambda_proj=lambda_proj,
        lambda_ortho=lambda_ortho,
        alpha_conflict=alpha_conflict,
        pair_mode=pair_mode,
        use_curv_exp=use_curv_exp,
        lambda_norm=args.lambda_norm,
        stat_pairs=stat_pairs,
        stat_modules=stat_modules,
    )
    out_dir = _output_dir(args)
    with open(osp.join(out_dir, 'merge_meta_stage3.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[Done] Stage-3 融合完成: 模块数={stat_modules}, 神经元对数={stat_pairs}")

# ---------------- Saving (复用 fam_merge 逻辑的简化版本) ----------------

def _output_dir(args) -> str:
    base_name = osp.basename(args.base_model.rstrip(os.sep))
    out_dir = osp.join(args.output_dir, base_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _save_model(args, merged_weights):
    print("\n正在保存融合模型 (Stage-3)...")
    output_dir = _output_dir(args)
    # 检测分片 index
    sft_index = osp.join(args.base_model, 'model.safetensors.index.json')
    bin_index = osp.join(args.base_model, 'pytorch_model.bin.index.json')

    def copy_side():
        for fn in os.listdir(args.base_model):
            if fn.endswith(('.json', '.model', '.py', '.md', '.txt')):
                src = osp.join(args.base_model, fn)
                dst = osp.join(output_dir, fn)
                if not osp.exists(dst):
                    try:
                        import shutil; shutil.copy(src, dst)
                    except Exception:
                        pass

    if osp.exists(sft_index):
        with open(sft_index, 'r') as f:
            idx = json.load(f)['weight_map']
        shard_map = defaultdict(dict)
        for k, v in merged_weights.items():
            if k in idx:
                shard_map[idx[k]][k] = v
        for shard, w in shard_map.items():
            safetensors.torch.save_file(w, osp.join(output_dir, shard))
        copy_side(); print(f"模型已保存 (safetensors 分片) -> {output_dir}"); return
    if osp.exists(bin_index):
        with open(bin_index, 'r') as f:
            idx = json.load(f)['weight_map']
        shard_map = defaultdict(dict)
        for k, v in merged_weights.items():
            if k in idx:
                shard_map[idx[k]][k] = v
        for shard, w in shard_map.items():
            torch.save(w, osp.join(output_dir, shard))
        copy_side(); print(f"模型已保存 (.bin 分片) -> {output_dir}"); return

    # 单文件
    sft_single = osp.join(args.base_model, 'model.safetensors')
    bin_single = osp.join(args.base_model, 'pytorch_model.bin')
    if osp.exists(sft_single):
        out_path = osp.join(output_dir, osp.basename(sft_single))
        safetensors.torch.save_file(merged_weights, out_path)
        copy_side(); print(f"模型已保存 (单一 safetensors) -> {out_path}"); return
    if osp.exists(bin_single):
        out_path = osp.join(output_dir, osp.basename(bin_single))
        torch.save(merged_weights, out_path)
        copy_side(); print(f"模型已保存 (单一 .bin) -> {out_path}"); return

    # 默认 safetensors
    out_path = osp.join(output_dir, 'model_stage3.safetensors')
    safetensors.torch.save_file(merged_weights, out_path)
    copy_side(); print(f"模型已保存 (默认 safetensors) -> {out_path}")

# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="TEFM Stage-3: 轨迹平坦干扰抑制融合")
    ap.add_argument('--base-model', type=str, required=True, help='模型A目录 (基础)')
    ap.add_argument('--donor-model', type=str, required=True, help='模型B目录 (捐赠)')
    ap.add_argument('--stage2', type=str, required=True, help='Stage-2 输出文件 (.pt)')
    ap.add_argument('--acts-base', type=str, default=None, help='模型A averaged activations (含 input / fai_H)')
    ap.add_argument('--acts-donor', type=str, default=None, help='模型B averaged activations')
    ap.add_argument('--base-lape', type=str, default=None, help='模型A LAPE sidecar')
    ap.add_argument('--donor-lape', type=str, default=None, help='模型B LAPE sidecar')
    ap.add_argument('--output-dir', type=str, required=True, help='输出目录根')
    # 权重控制
    ap.add_argument('--lambda-proj', type=float, default=1.0, help='投影分量系数')
    ap.add_argument('--lambda-ortho', type=float, default=0.8, help='正交分量系数')
    ap.add_argument('--lambda-norm', type=float, default=0.0, help='归一化层简单平均系数')
    ap.add_argument('--beta-flat', type=float, default=0.5, help='线性平坦惩罚系数 (1 - beta*H_B)')
    ap.add_argument('--beta-h', type=float, default=0.1, help='曲率差异指数惩罚系数 (exp(-beta_h*|hA-hB|))')
    ap.add_argument('--alpha-conflict', type=float, default=2.0, help='梯度冲突 sigmoid 放缩强度')
    ap.add_argument('--use-curv-exp', action='store_true', help='启用曲率差异指数惩罚 (叠乘)')
    ap.add_argument('--pair-mode', type=str, default='round_robin', choices=['round_robin','size-min','cartesian','centroid'], help='ensemble 展开为神经元对方式')
    ap.add_argument('--device', type=str, default=None, help='计算设备')
    # 新增控制项
    ap.add_argument('--deterministic', action='store_true', help='执行确定性合并 (关闭随机采样)')
    ap.add_argument('--no-random-pairs', action='store_true', help='与 --deterministic 作用类似，保证 pairs 不随机抽样')
    ap.add_argument('--max-row-change-ratio', type=float, default=0.25, help='单行 (神经元) 权重 L2 改变量最大占原始行范数比例')
    ap.add_argument('--module-max-change-ratio', type=float, default=0.35, help='单模块 (矩阵) 总改变量最大比例 (L2)')
    ap.add_argument('--conflict-mag-gamma', type=float, default=0.5, help='梯度幅值匹配抑制指数，0=关闭')
    return ap.parse_args()


def main():
    args = parse_args()
    tefm_stage3_merge(args)


if __name__ == '__main__':
    main()
