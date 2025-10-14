#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATAF Stage-1: Task Vector Metadata Extraction

目的:
  * 读取 (Base C, Model A, Model B) 的权重 (仅需要融合的 2D 线性层 + bias)。
  * 记录每个需要融合的参数的基本统计: 形状 / A,B,C 权重范数与任务向量范数 / 冲突角信息初步估计。
  * 不直接持久化完整 task vector (避免 7B 级模型巨大占用), Stage-2/3 将再次基于原始权重即时计算。

输出 (torch.save 到 --save):
{
  'meta': { base_model, model_a, model_b, datetime, total_params, linear_params, ... },
  'params': {
      param_key: {
         'shape': (d_out, d_in) or (n,),
         'is_bias': bool,
         'norm_A': float, 'norm_B': float, 'norm_C': float,
         'tauA_norm': float, 'tauB_norm': float,
         'tau_cos_AB': float or None (仅矩阵),
      }, ...
  }
}

后续:
  * Stage-2 读取该文件, 决定是否存储旋转矩阵 (维度阈值) 和缩放向量。
"""
from __future__ import annotations
import argparse, json, math, os, os.path as osp, datetime
from typing import Dict, Any
import torch
from tqdm import tqdm
from utils import load_weights, need_merge  # type: ignore

EPS = 1e-8

def _tau_stats(WA: torch.Tensor, WB: torch.Tensor, WC: torch.Tensor):
    tauA = WA - WC
    tauB = WB - WC
    tauA_norm = tauA.norm().item()
    tauB_norm = tauB.norm().item()
    cos = None
    if WA.ndim == 2:
        # 以列展开向量求整体余弦
        vA = tauA.flatten()
        vB = tauB.flatten()
        denom = (vA.norm()*vB.norm()).clamp_min(EPS)
        cos = (vA @ vB / denom).item()
    return dict(tauA_norm=tauA_norm, tauB_norm=tauB_norm, tau_cos_AB=cos)

def stage1(args: argparse.Namespace):
    print("\n--- [ATAF Stage-1: Task Vector Metadata Extraction] ---")
    weights_A = load_weights(args.model_a)
    weights_B = load_weights(args.model_b)
    weights_C = load_weights(args.base_model)

    params: Dict[str, Dict[str, Any]] = {}
    linear_cnt = 0
    bias_cnt = 0
    for k, WA in tqdm(weights_A.items(), desc='Scan Params'):
        if k not in weights_B or k not in weights_C:
            continue
        if not need_merge(k):
            continue
        WB = weights_B[k]
        WC = weights_C[k]
        if WA.shape != WB.shape or WA.shape != WC.shape:
            continue
        rec: Dict[str, Any] = {
            'shape': tuple(WA.shape),
            'is_bias': WA.ndim == 1,
            'norm_A': WA.float().norm().item(),
            'norm_B': WB.float().norm().item(),
            'norm_C': WC.float().norm().item(),
        }
        ts = _tau_stats(WA.float(), WB.float(), WC.float())
        rec.update(ts)
        params[k] = rec
        if WA.ndim == 2:
            linear_cnt += 1
        elif WA.ndim == 1:
            bias_cnt += 1

    meta = dict(
        base_model=args.base_model,
        model_a=args.model_a,
        model_b=args.model_b,
        datetime=str(datetime.datetime.now()),
        total_params=len(weights_A),
        recorded=len(params),
        linear_cnt=linear_cnt,
        bias_cnt=bias_cnt,
        store_task_vectors=False,
    )
    out = dict(meta=meta, params=params)
    os.makedirs(osp.dirname(args.save), exist_ok=True)
    torch.save(out, args.save)
    print(f"[Done] Saved Stage-1 metadata -> {args.save} (records={len(params)})")


def parse_args():
    ap = argparse.ArgumentParser(description='ATAF Stage-1 Task Vector Metadata Extraction')
    ap.add_argument('--model-a', required=True)
    ap.add_argument('--model-b', required=True)
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--save', required=True)
    return ap.parse_args()

if __name__ == '__main__':
    stage1(parse_args())
