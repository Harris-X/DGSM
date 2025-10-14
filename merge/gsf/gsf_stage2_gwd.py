#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSF-TEFM Stage-2: Gromov-Wasserstein Alignment & Psi Encoding (Enhanced)

新增改进 (2025-09-27):
 1. 支持 POT 库 (`ot`) 的精确 / 半精确 GWD 计算 (`--use-pot`)；优先使用 gromov_wasserstein2 (返回代价)；否则使用 gromov_wasserstein 获取耦合后自行再算 cost。
 2. 为自定义近似求解器加入 Early Stopping：监控 cost 相对改进 < tol 连续 patience 轮终止 (`--tol`, `--patience`).
 3. 记录每层 gwd_cost 与基于 (gamma, cost_scale) 估算出的融合 λ 值分布，便于 Stage-3 超参快速搜索。
 4. 修复原简化求解器导致所有层 π 相同的问题（之前的行/列重新缩放步骤强制回到均匀）。新近似器采用“GW -> cost 矩阵 M -> Sinkhorn (uniform marginals)”流程保留差异。

输出结构扩展: 每模块新增 'lambda_est' (若提供 gamma/cost-scale)，meta 中新增统计: cost_min/mean/max, lambda_min/mean/max。

若未安装 POT 或未指定 --use-pot，则退回近似实现；当 r 值较大 (>=256) 建议使用 POT 以免 O(r^4) 成本过高。
"""
from __future__ import annotations
import argparse
import math
import os
import os.path as osp
from typing import Dict, Tuple, Optional

import torch

_HAVE_POT = False
try:
    # POT 可能命名空间: ot.gromov
    import ot  # type: ignore
    from ot.gromov import gromov_wasserstein2, gromov_wasserstein  # type: ignore
    _HAVE_POT = True
except Exception:
    _HAVE_POT = False

EPS = 1e-8


def _load_subspaces(path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    obj = torch.load(path, map_location='cpu')
    subs = obj.get('subspaces', obj)
    return subs


def _softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max()
    e = (x - x_max).exp()
    return e / e.sum().clamp_min(EPS)


def _pairwise_sqdist(U: torch.Tensor, S: Optional[torch.Tensor] = None, mode: str = 'u') -> torch.Tensor:
    """根据子空间模式构造距离矩阵。
    mode:
      - 'u'  : 使用未加权左奇异向量列 (与原始实现一致, 可能退化为常量结构)
      - 'us' : 使用奇异值加权列向量 sqrt(S_i) * U[:,i] 以反映能量
      - 'usn': 同 'us' 但按向量范数再归一化 (降低尺度差距)
    返回: [r,r] 对称非负距离矩阵
    说明: 原实现所有列向量正交且范数=1 => pairwise dist=2 (off-diagonal) 造成 cost 退化。
          加入奇异值权重后可区分层间结构。
    """
    assert U.ndim == 2
    d_out, r = U.shape
    if mode not in ('u', 'us', 'usn'):
        mode = 'u'
    if mode == 'u' or S is None:
        X = U.T  # [r,d_out]
    else:
        # S: [r]; 权重幅值可能跨度大, 使用 sqrt 以减小动态范围
        w = torch.sqrt(S.clamp_min(EPS))  # [r]
        X = (U * w.unsqueeze(0)).T  # [r,d_out]
        if mode == 'usn':
            X = X / (X.norm(dim=1, keepdim=True).clamp_min(EPS))
    xx = (X * X).sum(dim=1, keepdim=True)
    dist = xx + xx.T - 2 * (X @ X.T)
    return dist.clamp_min(0.0)


def _sinkhorn_uniform(cost: torch.Tensor, reg: float, n_iter: int = 50) -> torch.Tensor:
    """简化 Sinkhorn: 在给定 cost 上求 uniform->uniform 的 OT plan (行列边际均匀)。"""
    n, m = cost.shape
    K = torch.exp(-cost / reg)  # [n,m]
    u = torch.full((n,), 1.0 / n, dtype=K.dtype)
    v = torch.full((m,), 1.0 / m, dtype=K.dtype)
    for _ in range(n_iter):
        u = (1.0 / n) / (K @ v).clamp_min(EPS)
        v = (1.0 / m) / (K.t() @ u).clamp_min(EPS)
    pi = torch.diag(u) @ K @ torch.diag(v)
    # 归一化以防数值漂移
    pi = pi / pi.sum().clamp_min(EPS)
    return pi


def _approx_gw(CA: torch.Tensor, CB: torch.Tensor, T: int, reg_sink: float, tol: float, patience: int, verbose: bool) -> Tuple[torch.Tensor, float, int]:
    """改进近似：
       1) 初始化 pi = uniform
       2) 构造四阶张量 L_{i j k l} = (CA_ij - CB_kl)^2 (按需分块)
       3) 每次迭代:  M_{i k} = sum_{j,l} L_{i j k l} * pi_{j l}
       4) 在 M 上做 Sinkhorn 得到新 pi
       5) 计算 cost，早停条件: 相对改进 < tol 连续 patience 次
    NOTE: O(r^4) 若 r 大需谨慎；r<=128 一般可接受。
    返回: (pi, cost, iters_used)
    """
    rA, rB = CA.shape[0], CB.shape[0]
    pi = torch.full((rA, rB), 1.0 / (rA * rB))
    prev_cost: Optional[float] = None
    bad_rounds = 0
    # 预构造差值张量分块策略: 直接广播 (可能占内存) -> 若超显存可后续优化
    CA_b = CA[:, :, None, None]
    CB_b = CB[None, None, :, :]
    L = (CA_b - CB_b).pow(2)  # [rA,rA,rB,rB]
    best_cost = float('inf')
    best_pi = pi.clone()
    for it in range(T):
        # M_{i,k}
        # (L * pi_jl) sum_{j,l}
        pi_b = pi[None, :, None, :]  # [1,rA,1,rB]
        M = (L * pi_b).sum(dim=(1, 3))  # [rA,rB]
        # Normalization for numerical stability
        M = M - M.min()
        pi = _sinkhorn_uniform(M, reg=reg_sink, n_iter=50)
        cost = _gwd_cost(CA, CB, pi)
        improved = cost < best_cost - 1e-9
        if improved:
            best_cost = cost
            best_pi = pi.clone()
        if verbose and (it % 5 == 0 or it == T - 1):
            tag = '*' if improved else ' '
            print(f"  [Iter {it:02d}] cost={cost:.6f}{tag}")
        if prev_cost is not None:
            rel = (prev_cost - cost) / max(EPS, prev_cost)
            if rel < tol:
                bad_rounds += 1
            else:
                bad_rounds = 0
            if bad_rounds >= patience:
                if verbose:
                    print(f"  [EarlyStop] it={it} rel_improve={rel:.3e}")
                return best_pi, best_cost, it + 1
        prev_cost = cost
    return best_pi, best_cost, T


def _gwd_align(CA: torch.Tensor, CB: torch.Tensor, args) -> Tuple[torch.Tensor, float, int]:
    """Wrapper: 使用 POT (若可用+开启) 否则用近似 GW。"""
    if args.use_pot and _HAVE_POT:
        rA, rB = CA.shape[0], CB.shape[0]
        p = torch.full((rA,), 1.0 / rA, dtype=torch.float64)
        q = torch.full((rB,), 1.0 / rB, dtype=torch.float64)
        CA_np = CA.double().cpu().numpy()
        CB_np = CB.double().cpu().numpy()
        try:
            # 优先 gromov_wasserstein2 (square_loss 默认) -> cost & coupling
            cost_val, pi_np = gromov_wasserstein2(CA_np, CB_np, p.numpy(), q.numpy(), 'square_loss', log=False, armijo=False, return_transport=True)
            pi = torch.from_numpy(pi_np).float()
            return pi, float(cost_val), -1
        except Exception:
            # 退回 gromov_wasserstein -> 需再算 cost
            try:
                pi_np = gromov_wasserstein(CA_np, CB_np, p.numpy(), q.numpy(), 'square_loss', log=False, armijo=False)
                pi = torch.from_numpy(pi_np).float()
                cost_val = _gwd_cost(CA, CB, pi)
                return pi, float(cost_val), -1
            except Exception as e:
                if args.verbose:
                    print(f"  [POT Fallback Fail] {type(e).__name__}: {e}; switching to approx.")
    # 近似 GW
    pi, cost, it_used = _approx_gw(CA, CB, T=args.iters, reg_sink=args.sink_reg, tol=args.tol, patience=args.patience, verbose=args.verbose)
    return pi, cost, it_used


def _gwd_cost(CA: torch.Tensor, CB: torch.Tensor, pi: torch.Tensor) -> float:
    # 计算四重和 (近似 O(r^4) 可能较大；使用分块降低成本)
    rA, rB = CA.shape[0], CB.shape[0]
    # 为保持简单，这里直接双重循环于 rA,rB (若 r<=128 成本仍可接受)；可进一步向量化
    cost = 0.0
    # 预计算 pi 外积结构: pi_ik * pi_jm
    for i in range(rA):
        CA_i = CA[i]  # [rA]
        pi_i = pi[i]  # [rB]
        for j in range(rA):
            dA = CA_i[j]
            pi_j = pi[j]
            # (dA - CB[k,m])^2 * pi_ik * pi_jm -> 展开
            # 向量化 k,m:
            # term = (dA - CB)^2
            diff2 = (dA - CB).pow(2)  # [rB,rB]
            contrib = (pi_i.unsqueeze(1) * pi_j.unsqueeze(0) * diff2).sum().item()
            cost += contrib
    return float(cost)


def compute_psi(S: torch.Tensor, gwd_cost: float) -> torch.Tensor:
    # S: [r]
    p = _softmax(S)  # 强度分布
    bar_s = (p * S).sum()
    bar_h = -(p * (p.clamp_min(EPS).log())).sum()
    bar_d = torch.tensor(float(gwd_cost))
    return torch.stack([bar_s, bar_h, bar_d])  # [3]


def _lambda_from_cost(cost: float, gamma: float, cost_scale: float) -> float:
    c_norm = min(1.0, cost / max(EPS, cost_scale))
    return 1.0 / (1.0 + math.exp(-gamma * (1.0 - c_norm)))


def stage2(args: argparse.Namespace):
    print("\n--- [GSF Stage-2: Gromov-Wasserstein Alignment & Psi Encoding] ---")
    if args.use_pot and not _HAVE_POT:
        print("[Warn] --use-pot 指定但未检测到 POT 库 (pip install POT)；使用近似求解器。")
    subsA = _load_subspaces(args.subs_a)
    subsB = _load_subspaces(args.subs_b)
    modules = sorted(set(subsA.keys()) & set(subsB.keys()))
    out_mod: Dict[str, Dict[str, torch.Tensor]] = {}
    costs = []
    lambdas = []
    for name in modules:
        blkA = subsA[name]; blkB = subsB[name]
        UA, SA = blkA['U'].float(), blkA['S'].float()
        UB, SB = blkB['U'].float(), blkB['S'].float()
        CA = _pairwise_sqdist(UA, SA, args.dist_mode)
        CB = _pairwise_sqdist(UB, SB, args.dist_mode)
        try:
            pi, cost, it_used = _gwd_align(CA, CB, args)
        except RuntimeError as e:
            print(f"[Warn] GWD align failed module {name}: {type(e).__name__}: {e}; skipping")
            continue
        psi_A = compute_psi(SA, cost)
        psi_B = compute_psi(SB, cost)
        rec = {
            'gwd_cost': torch.tensor(cost, dtype=torch.float32),
            'pi': pi.cpu(),
            'psi_A': psi_A.cpu(),
            'psi_B': psi_B.cpu(),
            'S_A': SA.cpu(), 'S_B': SB.cpu(),
            'rank_A': torch.tensor(SA.shape[0]),
            'rank_B': torch.tensor(SB.shape[0]),
            'iters_used': torch.tensor(it_used)
        }
        if args.gamma is not None and args.cost_scale is not None:
            lam_est = _lambda_from_cost(cost, args.gamma, args.cost_scale)
            rec['lambda_est'] = torch.tensor(lam_est, dtype=torch.float32)
            lambdas.append(lam_est)
        out_mod[name] = rec
        costs.append(cost)
        if args.verbose:
            add = f" lam={rec.get('lambda_est'):.4f}" if 'lambda_est' in rec else ""
            src = "POT" if (args.use_pot and _HAVE_POT) else "Approx"
            print(f"[GWD-{src}] {name}: cost={cost:.6f} rA={SA.shape[0]} rB={SB.shape[0]} it={it_used}{add}")
    # 汇总统计
    meta = {
        'subs_a': args.subs_a,
        'subs_b': args.subs_b,
        'iters': args.iters,
        'reg': args.reg,
        'use_pot': bool(args.use_pot and _HAVE_POT),
        'tol': args.tol,
        'patience': args.patience,
        'sink_reg': args.sink_reg,
        'gamma': args.gamma,
        'cost_scale': args.cost_scale,
        'cost_min': float(min(costs)) if costs else None,
        'cost_mean': float(sum(costs)/len(costs)) if costs else None,
        'cost_max': float(max(costs)) if costs else None,
        'lambda_min': float(min(lambdas)) if lambdas else None,
        'lambda_mean': float(sum(lambdas)/len(lambdas)) if lambdas else None,
        'lambda_max': float(max(lambdas)) if lambdas else None,
        'n_modules': len(out_mod)
    }
    torch.save({'modules': out_mod, 'meta': meta}, args.save)
    print(f"[Done] Stage-2 saved -> {args.save} modules={len(out_mod)}")
    if costs:
        print(f"  cost(min/mean/max)=({meta['cost_min']:.4f}/{meta['cost_mean']:.4f}/{meta['cost_max']:.4f})")
    if lambdas:
        print(f"  lambda(min/mean/max)=({meta['lambda_min']:.4f}/{meta['lambda_mean']:.4f}/{meta['lambda_max']:.4f})")


def parse_args():
    ap = argparse.ArgumentParser(description="GSF Stage-2: GWD alignment & psi encoding (POT + EarlyStop)")
    ap.add_argument('--subs-a', type=str, required=True)
    ap.add_argument('--subs-b', type=str, required=True)
    ap.add_argument('--save', type=str, required=True)
    # Solver knobs
    ap.add_argument('--use-pot', action='store_true', help='若安装 POT 则使用其 gromov_wasserstein2')
    ap.add_argument('--iters', type=int, default=30, help='最大迭代次数 (近似求解器)')
    ap.add_argument('--sink-reg', type=float, default=0.05, help='Sinkhorn 正则 (近似求解器内部 OT 步)')
    ap.add_argument('--tol', type=float, default=5e-4, help='Early stopping 相对改进阈值')
    ap.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    ap.add_argument('--reg', type=float, default=0.0, help='(deprecated) legacy regularization 保留占位')
    # Lambda estimation for logging
    ap.add_argument('--gamma', type=float, default=None, help='(可选) 估算 Stage-3 λ 的 gamma')
    ap.add_argument('--cost-scale', type=float, default=None, help='(可选) 估算 Stage-3 λ 的 cost_scale')
    ap.add_argument('--dist-mode', type=str, default='us', choices=['u','us','usn'], help='构造距离的点云模式: u(原始正交列) / us(奇异值加权) / usn(加权后归一)')
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


if __name__ == '__main__':
    stage2(parse_args())
