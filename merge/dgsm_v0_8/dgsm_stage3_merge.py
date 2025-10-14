#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DGSM-TEFM Stage-3: Dynamic-Adjusted Subspace Fusion Merge

与原 GSF Stage-3 基本相同, 区别:
  * 若 Stage-2 (DGSM) 中存在动态映射矩阵 M (r x r), 在重建 donor 差分投影时可选择使用对齐奇异方向加权。
  * 增加 --use-dynamic-m 选项: 启用时若记录 M 则先用 W_A 的 U_A 基底重投影，donor 差分投影方向按 M 的列混合。

保守实现: 为避免过度复杂化, 这里仍只对投影差分 τ_proj 进行缩放, 未改变正交残差处理。
"""
from __future__ import annotations
import argparse, json, math, os, os.path as osp
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch, safetensors.torch
from tqdm import tqdm
try:
    from .utils import load_weights, need_merge  # type: ignore
except Exception:
    from utils import load_weights, need_merge  # type: ignore

def _load_subspaces(path: str):
    try:
        obj = torch.load(path, map_location='cpu')
        return obj.get('subspaces', obj)
    except Exception:
        return None

EPS=1e-8

def _canon_param_key(param_key: str) -> str:
    k = param_key.replace('language_model.model.','model.').replace('language_model.','model.')
    if 'layers' in k:
        pos=k.find('layers'); k='model.'+k[pos:]
    return k

def _module_from_param_key(param_key:str)->str:
    k=_canon_param_key(param_key); parts=k.split('.')
    if len(parts)>=2: parts=parts[:-1]
    return '.'.join(parts)

def _alt_donor_key(donor:dict, base_key:str)->str | None:
    """在 donor 权重字典里为 base_key 寻找兼容命名的键。
    规则：
      1) 原样匹配 base_key
      2) 尝试在前面加 "language_model." 前缀（OneVision/QwenVL 常见命名）
    命中则返回匹配到的键，否则返回 None
    """
    if base_key in donor:
        return base_key
    alt = f"language_model.{base_key}"
    if alt in donor:
        return alt
    return None

def _svd_trunc(W:torch.Tensor, r:int):
    Wf=W.float(); r_use=min(r,Wf.shape[0],Wf.shape[1])
    if r_use<=0: return None
    try:
        U,S,Vh=torch.linalg.svd(Wf, full_matrices=False); return U[:,:r_use].contiguous(), S[:r_use].contiguous()
    except Exception: return None


def _softmax_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max() if x.numel() > 0 else x
    return torch.exp(x) / torch.exp(x).sum().clamp_min(EPS)


def _reuse_subspace(subspaces: Optional[Dict[str, Dict[str, torch.Tensor]]], name: str, r: int, rows: int) -> Optional[torch.Tensor]:
    if subspaces is None:
        return None
    blk = subspaces.get(name)
    if blk is None:
        return None
    U = blk.get('U') if isinstance(blk, dict) else None
    if U is None or not torch.is_tensor(U):
        return None
    if U.shape[0] != rows:
        return None
    r_use = min(r, U.shape[1])
    return U[:, :r_use].contiguous().float()


def _orthonormalize(U: torch.Tensor) -> torch.Tensor:
    if U.numel() == 0:
        return U
    try:
        q, _ = torch.linalg.qr(U, mode='reduced')
        return q
    except RuntimeError:
        return U


def _apply_mapping(Ub: torch.Tensor, M: Optional[torch.Tensor]) -> torch.Tensor:
    if M is None:
        return Ub
    return Ub @ M.T


def _compute_tfi_rel(delta_S: torch.Tensor, p_rel: torch.Tensor, psi_rel: torch.Tensor) -> Tuple[torch.Tensor, float]:
    bar_s = float(psi_rel[0]) if psi_rel.numel() >= 1 else float((p_rel * delta_S).sum().item())
    bar_h = float(psi_rel[1]) if psi_rel.numel() >= 2 else float(-(p_rel * (p_rel.clamp_min(EPS).log())).sum().item())
    bar_d = float(psi_rel[2]) if psi_rel.numel() >= 3 else 0.0
    bar_d = max(0.0, min(1.0, bar_d))
    bar_h = max(bar_h, EPS)
    tfi_base = abs(bar_s) * max(0.0, 1.0 - bar_d)
    tfi = p_rel * (tfi_base / bar_h) if tfi_base > 0 else torch.zeros_like(p_rel)
    return tfi, bar_d


def _build_tos(delta_S: torch.Tensor, p_rel: torch.Tensor, bar_d: float, beta: float) -> torch.Tensor:
    feat = torch.stack([
        delta_S,
        p_rel,
        torch.full_like(delta_S, bar_d)
    ], dim=1)
    norm = feat.norm(dim=1, keepdim=True).clamp_min(EPS)
    cos_sim = (feat @ feat.T) / (norm @ norm.T)
    return torch.relu(cos_sim) * math.exp(-beta * bar_d)


def _select_tfi_mask(scores: torch.Tensor, threshold: float, topk: int) -> torch.Tensor:
    mask = scores >= threshold if threshold > 0 else torch.ones_like(scores, dtype=torch.bool)
    if topk > 0:
        k = min(topk, scores.numel())
        top_idx = torch.topk(scores, k).indices
        top_mask = torch.zeros_like(scores, dtype=torch.bool)
        top_mask[top_idx] = True
        mask = mask | top_mask
    if mask.sum() == 0:
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[scores.argmax()] = True
    return mask


def _normalize_transport(pi: torch.Tensor, iters: int = 4) -> torch.Tensor:
    if pi.ndim != 2:
        return pi
    r, c = pi.shape
    if r == 0 or c == 0:
        return pi
    target_row = 1.0 / c
    target_col = 1.0 / r
    pi = pi / pi.sum().clamp_min(EPS)
    for _ in range(iters):
        row_sum = pi.sum(dim=1, keepdim=True).clamp_min(EPS)
        pi = pi * (target_row / row_sum)
        col_sum = pi.sum(dim=0, keepdim=True).clamp_min(EPS)
        pi = pi * (target_col / col_sum)
    return pi / pi.sum().clamp_min(EPS)


def _save_model(args, merged:Dict[str,torch.Tensor]):
    base_dir=osp.basename(args.base_model.rstrip(os.sep))
    out_root=osp.join(args.output_dir, base_dir, 'dgsm_merged'); os.makedirs(out_root, exist_ok=True)
    sft_index=osp.join(args.base_model,'model.safetensors.index.json')
    bin_index=osp.join(args.base_model,'pytorch_model.bin.index.json')
    def copy_side():
        for fn in os.listdir(args.base_model):
            if fn.endswith(('.json','.model','.py','.md')):
                try:
                    src=osp.join(args.base_model,fn); dst=osp.join(out_root,fn)
                    if not osp.exists(dst):
                        import shutil; shutil.copy(src,dst)
                except Exception: pass
    if osp.exists(sft_index):
        with open(sft_index,'r') as f: index=json.load(f)['weight_map']
        shards=defaultdict(dict)
        for k,v in merged.items():
            if k in index: shards[index[k]][k]=v
        for shard,sd in shards.items(): safetensors.torch.save_file(sd, osp.join(out_root,shard))
        copy_side(); print(f"[Save] Sharded safetensors -> {out_root}"); return
    if osp.exists(bin_index):
        with open(bin_index,'r') as f: index=json.load(f)['weight_map']
        shards=defaultdict(dict)
        for k,v in merged.items():
            if k in index: shards[index[k]][k]=v
        for shard,sd in shards.items(): torch.save(sd, osp.join(out_root,shard))
        copy_side(); print(f"[Save] Sharded .bin -> {out_root}"); return
    sft_single=osp.join(args.base_model,'model.safetensors')
    bin_single=osp.join(args.base_model,'pytorch_model.bin')
    if osp.exists(sft_single): safetensors.torch.save_file(merged, osp.join(out_root,'model.safetensors')); copy_side(); print(f"[Save] Single safetensors -> {out_root}"); return
    if osp.exists(bin_single): torch.save(merged, osp.join(out_root,'pytorch_model.bin')); copy_side(); print(f"[Save] Single .bin -> {out_root}"); return
    safetensors.torch.save_file(merged, osp.join(out_root,'model.safetensors')); copy_side(); print(f"[Save] Default safetensors -> {out_root}")


def dgsm_merge(args: argparse.Namespace):
    print("\n--- [DGSM Stage-3: Dynamic Subspace Fusion Merge] ---")
    weights_A=load_weights(args.base_model); weights_B=load_weights(args.donor_model)
    stage2=torch.load(args.stage2, map_location='cpu'); modules_info=stage2.get('modules', stage2)
    base_subs = _load_subspaces(args.base_subs) if args.base_subs else None
    donor_subs = _load_subspaces(args.donor_subs) if getattr(args, 'donor_subs', None) else None
    if base_subs is not None:
        print(f"[Info] Reusing Stage-1 base subspaces: {len(base_subs)} entries")
    if donor_subs is not None:
        print(f"[Info] Reusing Stage-1 donor subspaces: {len(donor_subs)} entries")
    merged=weights_A.copy(); stat_layers=stat_used=0
    missing_in_B=0; missing_examples=[]
    reuse_ok=reuse_fail=svd_fallback=0
    for k in tqdm(list(weights_A.keys()), desc='DGSM Merge'):
        if not need_merge(k): continue
        if k.endswith('.weight') and weights_A[k].ndim==2:
            mod=_module_from_param_key(k); blk=modules_info.get(mod)
            W_A=weights_A[k].float(); W_B=weights_B.get(k, None)
            if W_B is None:
                alt_k = _alt_donor_key(weights_B, k)
                if alt_k is None:
                    missing_in_B += 1
                    if len(missing_examples) < 10:
                        missing_examples.append(k)
                    continue
                W_B = weights_B[alt_k]
            W_B=W_B.float(); stat_layers+=1
            if blk is not None:
                r = int(blk.get('rank_A', torch.tensor(0)))
                if r <= 0:
                    continue
                U_A = _reuse_subspace(base_subs, mod, r, W_A.shape[0])
                if U_A is None:
                    if base_subs is not None:
                        reuse_fail += 1
                    svdA = _svd_trunc(W_A, r)
                    if svdA is None:
                        continue
                    U_A, _ = svdA
                    svd_fallback += 1
                else:
                    reuse_ok += 1

                U_B = _reuse_subspace(donor_subs, mod, r, W_B.shape[0])
                if U_B is None:
                    svdB = _svd_trunc(W_B, r)
                    if svdB is None:
                        U_B = U_A
                    else:
                        U_B, _ = svdB

                M_tensor = blk.get('M') if (args.use_dynamic_m and ('M' in blk)) else None
                M_tensor = M_tensor.float() if isinstance(M_tensor, torch.Tensor) else None
                if M_tensor is not None and M_tensor.shape[0] != r:
                    M_tensor = M_tensor[:r, :r]

                U_B_prime = _apply_mapping(U_B, M_tensor)
                U_rel = _orthonormalize((U_A + U_B_prime) / 2.0)

                delta_S = blk.get('delta_S') if 'delta_S' in blk else (blk['S_B'] - blk['S_A'])
                delta_S = delta_S.float()
                p_rel = blk.get('p_rel')
                p_rel = p_rel.float() if isinstance(p_rel, torch.Tensor) else _softmax_tensor(delta_S.abs())
                psi_rel = blk.get('psi_rel')
                if psi_rel is None:
                    psi_rel = torch.stack([
                        (p_rel * delta_S).sum(),
                        -(p_rel * (p_rel.clamp_min(EPS).log())).sum(),
                        torch.tensor(float(blk.get('bar_d', 0.0)), dtype=delta_S.dtype)
                    ])
                else:
                    psi_rel = psi_rel.float()

                tfi, bar_d = _compute_tfi_rel(delta_S, p_rel, psi_rel)
                mask = _select_tfi_mask(tfi, float(getattr(args, 'tfi_threshold', 0.0)), int(getattr(args, 'tfi_topk', 0)))
                if mask.sum() == 0:
                    mask = torch.ones_like(tfi, dtype=torch.bool)

                beta = float(getattr(args, 'mapping_beta', 0.1))
                tos = _build_tos(delta_S, p_rel, bar_d, beta)
                pi = blk['pi'].float()
                pi_adj = _normalize_transport((pi * tos).clamp_min(EPS))
                if M_tensor is not None:
                    pi_effective = M_tensor @ pi_adj
                else:
                    pi_effective = pi_adj

                aligned = pi_effective @ delta_S
                tfi_filtered = tfi.clone()
                tfi_filtered[~mask] = 0.0
                if tfi_filtered.sum() > 0:
                    aligned = aligned * (tfi_filtered / tfi_filtered.sum().clamp_min(EPS))

                tau = W_B - W_A
                # 投影到相对子空间: 先在 U_rel 基底下表示 tau，再按 aligned 缩放后回投影。
                tau_coords = U_rel.T @ tau  # [r, cols]
                tau_proj = U_rel @ (aligned.to(U_rel.dtype).unsqueeze(-1) * tau_coords)
                tau_ortho = tau - tau_proj

                if mask.any():
                    tfi_sel = tfi[mask]
                    tfi_sel = tfi_sel / tfi_sel.max().clamp_min(EPS)
                    mean_tfi = float(tfi_sel.mean().item())
                else:
                    mean_tfi = 0.0
                lam_rgsp = (1.0 / (1.0 + math.exp(bar_d))) * max(0.0, min(1.0, mean_tfi))
                lam_rgsp = max(0.0, min(1.0, lam_rgsp))

                W_avg = 0.5 * (W_A + W_B)
                W_new = W_avg + lam_rgsp * tau_proj + (1.0 - lam_rgsp) * tau_ortho
                merged[k]=W_new.to(weights_A[k].dtype); stat_used+=1
            else:
                alpha=float(args.fallback_alpha); merged[k]=(1-alpha)*W_A + alpha*W_B
        elif k.endswith('.bias') and weights_A[k].ndim==1:
            W_A=weights_A[k].float(); W_B=weights_B.get(k, None)
            if W_B is None:
                alt_k = _alt_donor_key(weights_B, k)
                if alt_k is None:
                    continue
                W_B = weights_B[alt_k]
            W_B=W_B.float(); alpha=float(args.bias_alpha)
            merged[k]=(1-alpha)*W_A + alpha*W_B
    _save_model(args, merged)
    meta=dict(base_model=args.base_model, donor_model=args.donor_model, stage2=args.stage2,
              fallback_alpha=float(args.fallback_alpha), bias_alpha=float(args.bias_alpha),
              use_dynamic_m=bool(args.use_dynamic_m), stat_layers=stat_layers, stat_used=stat_used,
              base_subs=args.base_subs, donor_subs=getattr(args, 'donor_subs', None),
              reuse_ok=reuse_ok, reuse_fail=reuse_fail, svd_fallback=svd_fallback,
              tfi_threshold=float(getattr(args, 'tfi_threshold', 0.0)),
              tfi_topk=int(getattr(args, 'tfi_topk', 0)),
              mapping_beta=float(getattr(args, 'mapping_beta', 0.1)),
              fusion_scheme='rgsp-tefm')
    base_dir=osp.basename(args.base_model.rstrip(os.sep))
    out_root=osp.join(args.output_dir, base_dir, 'dgsm_merged'); os.makedirs(out_root, exist_ok=True)
    with open(osp.join(out_root,'merge_meta_dgsm.json'),'w') as f: json.dump(meta,f,indent=2)
    print(f"[Done] DGSM merge complete: layers={stat_layers}, used={stat_used}, reuse_U={reuse_ok}, svd_fallback={svd_fallback}, reuse_fail={reuse_fail}")
    if missing_in_B>0:
        print(f"  [Warn] {missing_in_B} 处权重在 donor 中未找到（已跳过）。示例: {missing_examples}")


def parse_args():
    ap=argparse.ArgumentParser(description='DGSM Stage-3 Merge')
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--donor-model', required=True)
    ap.add_argument('--stage2', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--fallback-alpha', type=float, default=0.5)
    ap.add_argument('--bias-alpha', type=float, default=0.5)
    ap.add_argument('--use-dynamic-m', action='store_true')
    ap.add_argument('--base-subs', type=str, default=None, help='Stage-1 base 子空间文件 (含 U)，提供可跳过重复 SVD')
    ap.add_argument('--donor-subs', type=str, default=None, help='Stage-1 donor 子空间文件 (含 U)，用于避免重复 SVD')
    ap.add_argument('--tfi-threshold', type=float, default=0.0, help='Localization 阶段的 TFI 阈值 (DGSM-TEFM_v0_5.md §4)')
    ap.add_argument('--tfi-topk', type=int, default=0, help='Localization 阶段仅保留前 k 个 TFI 分量 (0 表示保留全部)')
    ap.add_argument('--mapping-beta', type=float, default=0.1, help='Mapping 阶段 TOS 权重中的 β 系数 (e^{-β \bar{d}_l})')
    return ap.parse_args()

if __name__=='__main__':
    dgsm_merge(parse_args())
