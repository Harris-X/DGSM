#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATAF Stage-2: Per-Group (Per-Param) Orthogonal + Diagonal Scaling Alignment

读取 Stage-1 元数据 + (A,B,C) 权重, 对每个 2D 线性层的任务向量:
  tau_A = W_A - W_C  (d_out x d_in)
  tau_B = W_B - W_C

我们从输出维度角度(行空间)做列方向的 Procrustes: 为便于解析解, 将矩阵视作 (d_out, d_in) 列向量集合。
具体步骤:
 1. 做列截断: 若 d_in > max_rank 则仅取前 max_rank 列 (近似, 主要为了内存和时间)。
 2. 构造 M = (tau_B^T @ tau_A) -> SVD = U Σ V^T -> P = U V^T
 3. 旋转后 tau_B_rot = tau_B @ P
 4. 列独立缩放: lambda_i = <tau_B_rot[:,i], tau_A[:,i]> / ||tau_B_rot[:,i]||^2
 5. 保存: P (d_in x d_in, 可能截断), lambda (d_in), 以及度量 (cos_after, energy_ratio, etc.)

注意: 对称性考虑我们不对 P 做强制 det=1 修正，因为纯正交包含反射不影响列内积对齐目标。
输出:
  torch.save -> {
    'meta': {...},
    'align': { param_key: { 'P': Tensor, 'lambda': Tensor, 'shape':(d_out,d_in), 'rank_cols': r_used,
                             'pre_cos': float, 'post_cos': float } }
  }

Stage-3 利用这些对齐信息进行任务向量融合。
"""
from __future__ import annotations
import argparse, os, os.path as osp, datetime, math
from typing import Dict, Any
import torch
from tqdm import tqdm
from utils import load_weights, need_merge  # type: ignore

EPS=1e-8

def _colwise_lambda(tA: torch.Tensor, tB_rot: torch.Tensor):
    # tA,tB_rot: [d_out, d_in]
    num = (tB_rot * tA).sum(dim=0)            # [d_in]
    denom = (tB_rot * tB_rot).sum(dim=0).clamp_min(EPS)
    lam = num / denom
    return lam

def stage2(args: argparse.Namespace):
    print("\n--- [ATAF Stage-2: Orthogonal + Diagonal Scaling Alignment] ---")
    meta1 = torch.load(args.stage1, map_location='cpu')
    weights_A = load_weights(args.model_a)
    weights_B = load_weights(args.model_b)
    weights_C = load_weights(args.base_model)

    align: Dict[str, Dict[str, Any]] = {}
    saved = 0
    for k, rec in tqdm(meta1['params'].items(), desc='Align Params'):
        shape = tuple(rec['shape'])
        if len(shape) != 2:
            continue  # 忽略 bias
        if k not in weights_A or k not in weights_B or k not in weights_C:
            continue
        WA = weights_A[k].float(); WB = weights_B[k].float(); WC = weights_C[k].float()
        if WA.shape != WB.shape or WA.shape != WC.shape:
            continue
        if not need_merge(k):
            continue
        d_out, d_in = WA.shape
        tauA = WA - WC
        tauB = WB - WC
        # 可选列截断
        r_cols = min(d_in, args.max_cols)
        if r_cols < d_in:
            tauA_use = tauA[:, :r_cols].contiguous()
            tauB_use = tauB[:, :r_cols].contiguous()
        else:
            tauA_use = tauA; tauB_use = tauB
        # Procrustes on column space: tauB_use^T tauA_use -> (r_cols x r_cols)
        M = tauB_use.T @ tauA_use  # r_cols x r_cols
        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            P = U @ Vh  # r_cols x r_cols
        except RuntimeError:
            # 回退单位
            P = torch.eye(r_cols, dtype=tauA.dtype)
        tauB_rot = tauB_use @ P  # [d_out, r_cols]
        lam = _colwise_lambda(tauA_use, tauB_rot)  # [r_cols]
        tauB_aligned = tauB_rot * lam.unsqueeze(0)
        # 计算对齐前/后整体余弦
        pre_cos = rec.get('tau_cos_AB', None)
        vA = tauA_use.flatten(); vB = tauB_use.flatten(); vB_al = tauB_aligned.flatten()
        post_cos = float((vA @ vB_al) / (vA.norm()*vB_al.norm()).clamp_min(EPS)) if vA.norm()>0 and vB_al.norm()>0 else None
        align[k] = {
            'P': P.cpu(),
            'lambda': lam.cpu(),
            'shape': (d_out, d_in),
            'rank_cols': r_cols,
            'pre_cos': pre_cos,
            'post_cos': post_cos,
        }
        saved += 1
    meta = dict(
        stage1=args.stage1,
        model_a=args.model_a,
        model_b=args.model_b,
        base_model=args.base_model,
        datetime=str(datetime.datetime.now()),
        max_cols=args.max_cols,
        saved=saved,
    )
    out = dict(meta=meta, align=align)
    os.makedirs(osp.dirname(args.save), exist_ok=True)
    torch.save(out, args.save)
    print(f"[Done] Stage-2 saved align info -> {args.save} (matrices={saved})")


def parse_args():
    ap = argparse.ArgumentParser(description='ATAF Stage-2 Alignment')
    ap.add_argument('--stage1', required=True)
    ap.add_argument('--model-a', required=True)
    ap.add_argument('--model-b', required=True)
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--save', required=True)
    ap.add_argument('--max-cols', type=int, default=4096, help='列截断上限 (防止极大 d_in 过慢)')
    return ap.parse_args()

if __name__ == '__main__':
    stage2(parse_args())
