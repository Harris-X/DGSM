#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSF-TEFM Stage-2b: 神经元群 (输出通道) 定位 与 行级 λ 估计

目的:
  基于 Stage-1 子空间 (U, S) 与 Stage-2 对齐 (pi, gwd_cost) 输出每层的行级(输出神经元)重要度与聚类分组, 为 Stage-3 提供更细粒度融合控制。

输入:
  --subs-a    Stage-1 A 子空间文件 (含 subspaces: {module: {'U','S','rank_used','shape'}} )
  --subs-b    Stage-1 B 子空间文件
  --stage2    Stage-2 输出 (含 modules: {module: {'pi','gwd_cost','S_A','S_B',...}} )

输出(.pt):
  {
    'modules': {
        module: {
           'row_lambda': FloatTensor[d_out],  # 每个输出神经元的 λ_proj (用于 τ_proj 逐行缩放)
           'group_id':  LongTensor[d_out],    # 聚类分组 id
           'group_lambda': FloatTensor[K],    # 每个组的 λ (投影部分)
           'gwd_cost': float,
           'rank_A': int,
           'rank_B': int,
           'K': int
        },...
    },
    'meta': {...}
  }

核心思路:
  1. 行嵌入: E_A = U_A * sqrt(S_A) (d_out x rA) 捕捉每个输出单元在子空间能量分布。
  2. 行能量: e_i = ||E_A[i]||_2^2.
  3. 对 E_A 做 KMeans (K 由 --k) 得到 group_id。
  4. 利用 Stage-2 pi (rA x rB) 估计 donor 对齐奇异值: S_B_aligned = pi * S_B。
     计算 A 基底方向相对能量比: ratio_j = S_B_aligned_j / (S_A_j + eps)。
  5. 投影每行的方向权重 w_i = sum_j (E_A[i,j]^2 / (sum_j E_A[i,j]^2 + eps)) * ratio_j。
  6. 组内聚合: group_ratio_g = mean_{i in g} w_i；再结合 gwd_cost 生成组 λ:
        cost_norm = min(1, gwd_cost / cost_scale)
        base_lambda = sigmoid( gamma * (1 - cost_norm) )
        lambda_g = base_lambda * (group_ratio_g)^{alpha}
     (若 group_ratio_g 过小, 加 eps 抑制数值问题)
  7. 行级 λ: row_lambda_i = lambda_{group_i} * w_i^{beta}

参数:
  --k 聚类簇数
  --gamma, --cost-scale, --alpha, --beta 控制 λ 形状

在 Stage-3 中: 若提供该文件, 对 τ_proj 行进行逐行缩放: τ_proj_scaled[i,:] = row_lambda[i] * τ_proj[i,:]

不依赖外部库 (自实现简单 KMeans)。
"""
from __future__ import annotations
import argparse
import math
import os
import os.path as osp
from typing import Dict, Tuple

import torch

EPS = 1e-8


def _load_subspaces(path: str):
    obj = torch.load(path, map_location='cpu')
    return obj.get('subspaces', obj)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def _kmeans(x: torch.Tensor, k: int, iters: int = 50, seed: int = 42):
    """简易 KMeans (L2) 返回 (centroids, labels). x: [n,d]."""
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    n, d = x.shape
    if k >= n:
        labels = torch.arange(n) % k
        return x.clone()[:k], labels
    # 初始化: 随机选择 k 个点
    perm = torch.randperm(n, generator=g)
    cent = x[perm[:k]].clone()
    for _ in range(iters):
        # 分配
        dist = (x.unsqueeze(1) - cent.unsqueeze(0)).pow(2).sum(-1)  # [n,k]
        labels = dist.argmin(dim=1)
        # 更新
        new_cent = torch.zeros_like(cent)
        counts = torch.zeros(k, dtype=torch.long)
        for i in range(k):
            mask = (labels == i)
            if mask.any():
                new_cent[i] = x[mask].mean(dim=0)
                counts[i] = mask.sum()
            else:
                # 重新随机补
                new_cent[i] = x[perm[torch.randint(0, n, (1,), generator=g)]]
                counts[i] = 1
        if torch.allclose(new_cent, cent, atol=1e-5):
            cent = new_cent
            break
        cent = new_cent
    return cent, labels


def _row_embeddings(U: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    # E = U * sqrt(S)  (broadcast S over rows)
    w = torch.sqrt(S.clamp_min(EPS))  # [r]
    return U * w.unsqueeze(0)  # [d_out, r]


def compute_group_lambda(E_A: torch.Tensor, S_A: torch.Tensor, S_B: torch.Tensor, pi: torch.Tensor,
                         gwd_cost: float, args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 输入 shapes: U已不需要, E_A [d_out,rA]; S_A[rA], S_B[rB], pi[rA,rB]
    rA = S_A.shape[0]
    # donor 对齐奇异值: [rA]
    S_B_aligned = (pi * S_B.unsqueeze(0)).sum(dim=1)  # sum_k pi_{jk} * S_B[k]
    ratio = S_B_aligned / (S_A + EPS)  # [rA]
    ratio = ratio.clamp_min(0.)
    # 行方向权重: E_A[i,j]^2 占比 * ratio_j
    energy = (E_A * E_A)  # [d_out,rA]
    row_energy_sum = energy.sum(dim=1, keepdim=True).clamp_min(EPS)
    weight_frac = energy / row_energy_sum  # 每行对每基底的能量占比
    w_row = (weight_frac * ratio.unsqueeze(0)).sum(dim=1)  # [d_out]
    # 聚类 (在 E_A 上或使用 w_row? 采用 E_A 捕捉分布结构)
    cent, labels = _kmeans(E_A, args.k, iters=args.kmeans_iters, seed=args.seed)
    K = args.k
    group_ratio = torch.zeros(K)
    for g in range(K):
        mask = labels == g
        if mask.any():
            group_ratio[g] = w_row[mask].mean()
        else:
            group_ratio[g] = w_row.mean()
    # 基础 λ (层级):
    cost_scale = max(EPS, float(args.cost_scale))
    cost_norm = min(1.0, gwd_cost / cost_scale)
    base_lambda = float(_sigmoid(torch.tensor(args.gamma * (1.0 - cost_norm))))
    alpha = float(args.alpha)
    beta = float(args.beta)
    group_lambda = base_lambda * (group_ratio.clamp_min(EPS) ** alpha)
    row_lambda = group_lambda[labels] * (w_row.clamp_min(EPS) ** beta)
    # 归一尺度防爆: 限制在 [0,1]
    row_lambda = row_lambda.clamp(0.0, 1.0)
    group_lambda = group_lambda.clamp(0.0, 1.0)
    return row_lambda, group_lambda, labels


def stage2b(args: argparse.Namespace):
    print("\n--- [GSF Stage-2b: Neuron Group Localization] ---")
    subsA = _load_subspaces(args.subs_a)
    subsB = _load_subspaces(args.subs_b)
    stage2 = torch.load(args.stage2, map_location='cpu')
    modules2 = stage2.get('modules', stage2)

    modules = sorted(set(subsA.keys()) & set(subsB.keys()) & set(modules2.keys()))
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in modules:
        blkA = subsA[name]; blkB = subsB[name]; blk2 = modules2[name]
        U_A, S_A = blkA['U'].float(), blkA['S'].float()
        S_B = blk2['S_B'].float() if 'S_B' in blk2 else blkB['S'].float()
        pi = blk2['pi'].float()
        gwd_cost = float(blk2['gwd_cost'])
        # 行嵌入
        E_A = _row_embeddings(U_A, S_A)
        try:
            row_lambda, group_lambda, labels = compute_group_lambda(E_A, S_A, S_B, pi, gwd_cost, args)
        except RuntimeError as e:
            print(f"[Warn] group compute fail {name}: {e}")
            continue
        out[name] = {
            'row_lambda': row_lambda.cpu(),
            'group_id': labels.cpu(),
            'group_lambda': group_lambda.cpu(),
            'gwd_cost': torch.tensor(gwd_cost),
            'rank_A': torch.tensor(int(S_A.shape[0])),
            'rank_B': torch.tensor(int(S_B.shape[0])),
            'K': torch.tensor(int(args.k))
        }
        if args.verbose:
            print(f"[LOC] {name}: cost={gwd_cost:.4f} base_groups={args.k} rowλ(mean)={row_lambda.mean():.4f}")
    meta = {
        'subs_a': args.subs_a,
        'subs_b': args.subs_b,
        'stage2': args.stage2,
        'k': args.k,
        'gamma': args.gamma,
        'cost_scale': args.cost_scale,
        'alpha': args.alpha,
        'beta': args.beta,
        'kmeans_iters': args.kmeans_iters,
        'seed': args.seed,
        'n_modules': len(out)
    }
    torch.save({'modules': out, 'meta': meta}, args.save)
    print(f"[Done] Stage-2b saved -> {args.save} modules={len(out)}")


def parse_args():
    ap = argparse.ArgumentParser(description="GSF Stage-2b: Neuron group localization & row-level lambda")
    ap.add_argument('--subs-a', type=str, required=True)
    ap.add_argument('--subs-b', type=str, required=True)
    ap.add_argument('--stage2', type=str, required=True)
    ap.add_argument('--save', type=str, required=True)
    ap.add_argument('--k', type=int, default=8, help='KMeans 簇个数')
    ap.add_argument('--kmeans-iters', type=int, default=50)
    ap.add_argument('--gamma', type=float, default=4.0, help='基础层级 λ 的 gamma')
    ap.add_argument('--cost-scale', type=float, default=1.0, help='gwd cost 归一尺度')
    ap.add_argument('--alpha', type=float, default=0.5, help='组级 ratio 幂次')
    ap.add_argument('--beta', type=float, default=0.5, help='行级 w_row 幂次')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


if __name__ == '__main__':
    stage2b(parse_args())
