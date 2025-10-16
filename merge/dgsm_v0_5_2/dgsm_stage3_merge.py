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
from typing import Dict, Tuple
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

def _strip_multiway_segment(name: str) -> str:
    if '.multiway.' not in name:
        return name
    parts = name.split('.')
    out = []
    skip_next = False
    for part in parts:
        if skip_next:
            skip_next = False
            continue
        if part == 'multiway':
            skip_next = True
            continue
        out.append(part)
    return '.'.join(out)

def _alt_donor_key(donor:dict, base_key:str)->str | None:
    """在 donor 权重字典里为 base_key 寻找兼容命名的键。
    规则：
      1) 原样匹配 base_key
      2) 尝试在前面加 "language_model." 前缀（OneVision/QwenVL 常见命名）
      3) 若包含 multiway.* 结构，尝试去除 multiway.<idx>
    命中则返回匹配到的键，否则返回 None
    """
    candidates = [base_key]
    stripped = _strip_multiway_segment(base_key)
    if stripped != base_key:
        candidates.append(stripped)
    # 前缀互换
    more = []
    for cand in list(candidates):
        if cand.startswith('language_model.'):
            core = cand[len('language_model.') :]
            more.append(core)
        else:
            more.append(f'language_model.{cand}')
    for cand in more:
        if cand not in candidates:
            candidates.append(cand)

    # 去除 multiway 后再拼接 language_model 前缀（避免重复）
    stripped_lang = _strip_multiway_segment(f'language_model.{base_key}')
    if stripped_lang not in candidates:
        candidates.append(stripped_lang)

    for cand in candidates:
        if cand in donor:
            return cand
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


def _compute_tfi(
    S: torch.Tensor,
    psi: torch.Tensor,
    beta: float,
    entropy_eps: float,
) -> Tuple[torch.Tensor, float, float, float]:
    """Compute Task Fusion Importance scores with structured alignment decay."""
    S_vec = S.view(-1).float()
    if psi.numel() >= 1:
        bar_s = float(psi[0])
    else:
        probs = _softmax_tensor(S_vec)
        bar_s = float((probs * S_vec).sum().item())
    if psi.numel() >= 2:
        bar_h = float(psi[1])
    else:
        probs = _softmax_tensor(S_vec)
        bar_h = float(-(probs * probs.clamp_min(EPS).log()).sum().item())
    if psi.numel() >= 3:
        bar_d = float(psi[2])
    else:
        bar_d = 0.0

    denom = max(entropy_eps, bar_h + entropy_eps)
    decay = math.exp(-max(beta, 0.0) * max(bar_d, 0.0))
    tfi = (S_vec.clamp_min(0.0) / denom) * decay
    tfi = tfi.clamp_min(0.0)
    return tfi, bar_d, bar_h, bar_s


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
    if base_subs is not None:
        print(f"[Info] Reusing Stage-1 base subspaces: {len(base_subs)} entries")
    merged=weights_A.copy(); stat_layers=stat_used=0
    missing_in_B=0; missing_examples=[]
    reuse_ok=reuse_fail=svd_fallback=0
    for k in tqdm(list(weights_A.keys()), desc='DGSM Merge'):
        if not need_merge(k): continue
        if k.endswith('.weight') and weights_A[k].ndim==2:
            mod=_module_from_param_key(k)
            module_lookup_key = mod
            blk=modules_info.get(module_lookup_key)
            if blk is None and '.multiway.' in module_lookup_key:
                alt_mod = _strip_multiway_segment(module_lookup_key)
                blk = modules_info.get(alt_mod)
                if blk is not None:
                    module_lookup_key = alt_mod
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
                gwd_cost = float(blk.get('gwd_cost', 0.0))
                psiA_tensor = blk.get('psi_A', None)
                if isinstance(psiA_tensor, torch.Tensor) and psiA_tensor.numel() >= 3:
                    bar_d = float(psiA_tensor[2])
                else:
                    bar_d = gwd_cost

                lambda_beta = float(args.lambda_beta) if args.lambda_beta is not None else float(args.tfi_beta)
                lam_candidates = []
                if bool(getattr(args, 'use_lambda_est', False)) and ('lambda_est' in blk):
                    lam_stage2 = float(blk['lambda_est'])
                    if getattr(args, 'lam_scale', None) is not None:
                        lam_stage2 *= float(args.lam_scale)
                    lam_candidates.append(max(0.0, min(1.0, lam_stage2)))

                # 基于 Stage-2 的 psi_A 熵自适应调整正交项权重（越确定性，越小的正交注入）
                base_ortho=float(args.ortho_scale)
                if bool(getattr(args, 'ortho_adapt', False)) and ('psi_A' in blk):
                    psiA = blk['psi_A']  # [bar_s, bar_h, bar_d]
                    bar_h = float(psiA[1]) if torch.is_tensor(psiA) else float(psiA[1])
                    # 归一化熵: 经验范围 ~ [0, ln(r)]，用 r 的对数进行归一
                    r_eff = int(blk.get('rank_A', torch.tensor(1)))
                    import math as _m
                    h_max = max(1e-6, _m.log(max(2, r_eff)))
                    h_norm = min(1.0, max(0.0, bar_h / h_max))
                    # 低熵(更确定) -> 减小正交注入；高熵 -> 增大到 base_ortho
                    ortho_scale = base_ortho * h_norm
                else:
                    ortho_scale=base_ortho
                r=int(blk['rank_A'])
                U_A=None
                if base_subs is not None and mod in base_subs:
                    try:
                        U_full = base_subs[mod]['U']  # d_out x r_full
                        if U_full.shape[0] == W_A.shape[0]:
                            U_A = U_full[:, :r].contiguous().float(); reuse_ok += 1
                        else:
                            reuse_fail += 1
                    except Exception:
                        reuse_fail += 1
                if U_A is None:
                    svdA=_svd_trunc(W_A, r)
                    if svdA is None: continue
                    U_A,_=svdA; svd_fallback += 1
                tau = W_B - W_A
                coef = U_A.T @ tau  # r x d_in

                weights_vec = None
                has_tefm = all(key in blk for key in ('pi', 'S_A', 'S_B', 'psi_A', 'psi_B'))
                M_tensor = blk['M'].float() if (args.use_dynamic_m and ('M' in blk)) else None

                if has_tefm:
                    S_A = blk['S_A'].float().view(-1)
                    S_B = blk['S_B'].float().view(-1)
                    psi_A = blk['psi_A'].float().view(-1)
                    psi_B = blk['psi_B'].float().view(-1)
                    tfi, bar_d, bar_h, bar_s = _compute_tfi(
                        S_A,
                        psi_A,
                        float(args.tfi_beta),
                        float(args.entropy_eps),
                    )
                    mask = _select_tfi_mask(
                        tfi,
                        float(getattr(args, 'tfi_threshold', 0.0)),
                        int(getattr(args, 'tfi_topk', 0)),
                    )
                    weights_vec = tfi.clone()
                    if weights_vec.numel() > 0 and weights_vec.max() > 0:
                        weights_vec = weights_vec / weights_vec.max().clamp_min(EPS)
                    weights_vec = weights_vec * mask.float()
                    if weights_vec.max() <= 0:
                        weights_vec = mask.float()
                    # Mapping: 调整 π*_{k,p} ← π*_{k,p} · TOS_{k,p} / Z，其中 TOS 基于余弦相似和 e^{-β \bar{d}_l}
                    p_A = _softmax_tensor(S_A)
                    p_B = _softmax_tensor(S_B)
                    bar_d_B = float(psi_B[2]) if psi_B.numel() >= 3 else bar_d
                    psiA_comp = torch.stack([S_A, p_A, S_A.new_full(S_A.shape, bar_d)], dim=1)
                    psiB_comp = torch.stack([S_B, p_B, S_B.new_full(S_B.shape, bar_d_B)], dim=1)
                    normA = psiA_comp.norm(dim=1, keepdim=True).clamp_min(EPS)
                    normB = psiB_comp.norm(dim=1, keepdim=True).clamp_min(EPS)
                    cos_sim = (psiA_comp @ psiB_comp.T) / (normA @ normB.T)
                    mapping_beta = float(getattr(args, 'mapping_beta', 0.1))
                    dist_decay = math.exp(-mapping_beta * max(bar_d, 0.0))
                    tos = torch.relu(cos_sim) * dist_decay
                    pi = blk['pi'].float()
                    pi_adj = _normalize_transport((pi * tos).clamp_min(EPS))
                    if M_tensor is not None:
                        pi_effective = M_tensor @ pi_adj
                    else:
                        pi_effective = pi_adj
                    alignment = pi_effective @ S_B
                    if alignment.numel() > 0 and alignment.max() > 0:
                        alignment = alignment / alignment.max().clamp_min(EPS)
                        weights_vec = torch.max(weights_vec, alignment * mask.float())

                if M_tensor is not None:
                    coef = M_tensor @ coef

                if weights_vec is not None:
                    coef = coef * weights_vec.to(coef.dtype).unsqueeze(1)

                lam_doc = math.exp(-max(lambda_beta, 0.0) * max(bar_d, 0.0))
                lam_candidates.append(lam_doc)
                lam = sum(lam_candidates) / len(lam_candidates)
                lam = max(0.0, min(1.0, lam))

                tau_proj = U_A @ coef
                tau_ortho = tau - tau_proj
                tau_ortho_scaled = ortho_scale * tau_ortho
                W_new = W_A + lam * tau_proj + (1.0 - lam) * tau_ortho_scaled
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
              ortho_scale=float(args.ortho_scale),
              fallback_alpha=float(args.fallback_alpha), bias_alpha=float(args.bias_alpha),
              use_dynamic_m=bool(args.use_dynamic_m), stat_layers=stat_layers, stat_used=stat_used,
              base_subs=args.base_subs, reuse_ok=reuse_ok, reuse_fail=reuse_fail, svd_fallback=svd_fallback,
              tfi_threshold=float(getattr(args, 'tfi_threshold', 0.0)),
              tfi_topk=int(getattr(args, 'tfi_topk', 0)),
              tfi_beta=float(getattr(args, 'tfi_beta', 0.1)),
              lambda_beta=float(args.lambda_beta) if args.lambda_beta is not None else float(getattr(args, 'tfi_beta', 0.1)),
              entropy_eps=float(getattr(args, 'entropy_eps', EPS)),
              mapping_beta=float(getattr(args, 'mapping_beta', 0.1)))
    base_dir=osp.basename(args.base_model.rstrip(os.sep))
    out_root=osp.join(args.output_dir, base_dir, 'dgsm_merged'); os.makedirs(out_root, exist_ok=True)
    with open(osp.join(out_root,'merge_meta_dgsm.json'),'w') as f: json.dump(meta,f,indent=2)
    print(f"[Done] DGSM merge complete: layers={stat_layers}, used={stat_used}, reuse_U={reuse_ok}, svd_fallback={svd_fallback}, reuse_fail={reuse_fail}")
    if missing_in_B>0:
        print(f"  [Warn] {missing_in_B} 处权重在 donor 中未找到（已跳过）。示例: {missing_examples}")


def parse_args():
    ap=argparse.ArgumentParser(description='DGSM Stage-3 Merge')
    ap.add_argument('--base-model', default="downloaded_models/Qwen2-VL-7B-Instruct",required=True)
    ap.add_argument('--donor-model',default="downloaded_models/llava-onevision-qwen2-7b-si-hf", required=True)
    ap.add_argument('--stage2', default="work/dgsm/stage2_r128_us_on_gw_e0.5_it30_reg0.05_ds8.pt",required=True)
    ap.add_argument('--output-dir', default="dgsm_merged_models_stage3", required=True)
    ap.add_argument('--ortho-scale', type=float, default=0.3)
    ap.add_argument('--fallback-alpha', type=float, default=0.6)
    ap.add_argument('--bias-alpha', type=float, default=0.3)
    ap.add_argument('--use-dynamic-m', action='store_true')
    ap.add_argument('--base-subs', type=str, default=None, help='Stage-1 base 子空间文件 (含 U)，提供可跳过重复 SVD')
    ap.add_argument('--use-lambda-est', action='store_true', help='使用 Stage-2 中每层的 lambda_est 作为融合强度')
    ap.add_argument('--lam-scale', type=float, default=None, help='对 per-layer lambda_est 进行缩放的系数, 可微调融合强度')
    ap.add_argument('--ortho-adapt', action='store_true', help='根据 Stage-2 的 psi 熵自适应调整正交项注入强度')
    ap.add_argument('--tfi-threshold', type=float, default=0.0, help='TFI 掩码阈值，<=0 表示仅由 top-k 控制')
    ap.add_argument('--tfi-topk', type=int, default=0, help='Localization 阶段仅保留前 k 个 TFI 分量 (0 表示保留全部)')
    ap.add_argument('--tfi-beta', type=float, default=0.1, help='TFI 结构化对齐衰减参数 β')
    ap.add_argument('--lambda-beta', type=float, default=None, help='融合系数 λ 的距离衰减参数 (默认等于 --tfi-beta)')
    ap.add_argument('--entropy-eps', type=float, default=1e-6, help='计算 TFI 时的熵平滑项 ε')
    ap.add_argument('--mapping-beta', type=float, default=0.1, help='Mapping 阶段 TOS 权重中的 β 系数 (e^{-β \bar{d}_l})')
    return ap.parse_args()

if __name__=='__main__':
    dgsm_merge(parse_args())
