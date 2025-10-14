#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEFM Stage-2 (LAPE 增强版本): 轨迹导向的 Ensemble 定位与映射

输入：
  - 模型 A 与 模型 B 的激活缓存（Stage-1 生成的 averaged activations .pt）
  - 对应的 LAPE sidecar 文件（*_lape.pt），其中包含每个模块的 phi 向量：phi_a, phi_g, phi_L

目标：
  1. 读取并整合每个模块的神经元级特征：
       - phi_a: 激活期望（长度无关）
       - phi_g: 梯度轨迹近似
       - phi_L: 损失标量（广播）
       - 构造 Hess 对角近似: h_i = (phi_g_i)^2 + phi_L_i + eps
  2. 计算神经元轨迹-平坦重要性 (TFI)：
       TFI_i = |phi_g_i| * |phi_a_i| / (h_i + eps)
     （原理论使用 MI(phi_a, phi_L)，但 Stage-1 保存的是聚合后单点均值，无法估计互信息。此处以 |phi_a| 作为可行替代；可扩展为加载 per-sample 统计时再替换。）
  3. 基于特征向量 f_i = [norm(phi_a_i), norm(phi_g_i), norm(phi_L_i), norm(TFI_i)] 对神经元进行聚类 -> 得到若干 Ensemble (E_k)。默认使用 KMeans；若未安装 sklearn，则退化为阈值二分或全部单簇。
  4. 每个 Ensemble 聚合出其“代表向量” v_k = mean( [phi_a, phi_g, phi_L] ) 以及平均曲率 h_k = mean(h_i)。
  5. 在模型 A 与 B 的对应模块上，计算轨迹导向相似度 (TOS)：
        TOS(k,p) = cos( v^A_k , v^B_p ) * exp( -beta * | h^A_k - h^B_p | )
     并通过匈牙利算法（若可用）或贪心匹配获得 ensemble 对齐映射。

输出：
  - 一个 .pt 文件，结构：
      {
         'meta': {... 全局参数 ...},
         'modules': {
             module_name: {
                 'ensembles_A': LongTensor[H_A]  # 每个神经元所属簇 id（-1 表示未分配）
                 'ensembles_B': LongTensor[H_B]
                 'K_A': int, 'K_B': int,
                 'centroids_A': FloatTensor[K_A, 3],
                 'centroids_B': FloatTensor[K_B, 3],
                 'h_mean_A': FloatTensor[K_A],
                 'h_mean_B': FloatTensor[K_B],
                 'pairs': LongTensor[M, 2],      # (k_A, k_B)
                 'scores': FloatTensor[M],       # 对应 TOS 分数
                 'module_size_A': int,
                 'module_size_B': int,
                 'tfi_A': FloatTensor[H_A],      # 可选：保存每个神经元 TFI
                 'tfi_B': FloatTensor[H_B]
             }, ...
         }
      }

使用示例：
  python merge/TEFM/tefm_stage2_ensemble.py \
      --acts-a activations/mPLUG-Owl2_meta.pt \
      --acts-a-lape activations/mPLUG-Owl2_meta_lape.pt \
      --acts-b activations/llava_v1.5_7b_meta.pt \
      --acts-b-lape activations/llava_v1.5_7b_meta_lape.pt \
      --save activations/tefm_stage2_mPLUG-Owl2_TO_llava_v1.5_7b.pt \
      --k 8 --beta 0.1

注意：
  - 若未提供 *_lape.pt，将尝试自动用 <base>_lape.pt 补齐。
  - 若模块在任一模型缺失所需 phi 向量，则跳过该模块。
  - 该实现为最小可行版本，后续可扩展：
       * 自动 K 选择 (silhouette / elbow)
       * 替换 TFI 中的 MI 近似为真正互信息（需保存 per-sample phi）
       * 跨模块联合聚类 / 跨层对齐
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import Dict, Any, Tuple, List, Optional
import math
import warnings

import torch
import numpy as np

try:
    from sklearn.cluster import KMeans
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

EPS = 1e-8

# -----------------------------
# Helpers
# -----------------------------

def _canon_module_name(name: str) -> str:
    k = name.replace("language_model.model.", "model.").replace("language_model.", "model.")
    if "layers" in k:
        pos = k.find("layers")
        k = "model." + k[pos:]
    return k

def _load_pt(path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    d = torch.load(path, map_location="cpu")
    return { _canon_module_name(k): v for k, v in d.items() }


def _load_acts_and_lape(acts_path: str, lape_path: Optional[str]) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
    acts = _load_pt(acts_path)
    if lape_path is None:
        base, ext = osp.splitext(acts_path)
        cand = base + "_lape" + ext
        if osp.exists(cand):
            lape_path = cand
    lape: Dict[str, Dict[str, torch.Tensor]] = {}
    if lape_path and osp.exists(lape_path):
        lape = _load_pt(lape_path)
    else:
        warnings.warn(f"LAPE sidecar file not found for {acts_path}; modules will lack phi_* vectors.")
    return acts, lape


def _prepare_module_block(acts_blk: Dict[str, torch.Tensor], lape_blk: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
    """ Merge averaged activations + LAPE phi vectors into a unified dict.
        Expect lape_blk contains phi_a / phi_g / phi_L (1D tensors length H). Return None if insufficient data.
    """
    if lape_blk is None:
        return None
    if ("phi_a" not in lape_blk) or ("phi_g" not in lape_blk) or ("phi_L" not in lape_blk):
        return None
    out: Dict[str, torch.Tensor] = {}
    try:
        out['phi_a'] = lape_blk['phi_a'].detach().float().cpu()
        out['phi_g'] = lape_blk['phi_g'].detach().float().cpu()
        out['phi_L'] = lape_blk['phi_L'].detach().float().cpu()
    except Exception:
        return None
    Hs = {out['phi_a'].shape[0], out['phi_g'].shape[0], out['phi_L'].shape[0]}
    if len(Hs) != 1:
        return None
    # Optional extras from averaged activations / FAI statistics
    for k in ['output', 'input', 'fai', 'fai_A', 'fai_MI', 'fai_H']:
        if k in acts_blk and isinstance(acts_blk[k], torch.Tensor):
            out[k] = acts_blk[k].detach().float().cpu()
    return out


def _estimate_mi_from_samples(phi_a_samples: torch.Tensor, phi_L_samples: torch.Tensor, bins: int = 32) -> torch.Tensor:
    """Estimate per-neuron MI( phi_a , phi_L ) via discretization (histogram based).
    phi_a_samples: [N, H]; phi_L_samples: [N, H] (broadcasted loss samples per neuron)
    Returns Tensor[H] mutual information (non-negative).
    Simplified discrete MI: sum_{x,y} p(x,y) log ( p(x,y) / (p(x)p(y)) ).
    We independently bin each neuron's samples. For efficiency we vectorize over neurons using torch.histc sequentially.
    Note: This is an approximation; for large N consider kNN MI estimators.
    """
    N, H = phi_a_samples.shape
    # Normalize each neuron's samples to [0,1] for binning stability
    a = phi_a_samples
    L = phi_L_samples
    a_min = a.min(dim=0).values; a_max = a.max(dim=0).values
    L_min = L.min(dim=0).values; L_max = L.max(dim=0).values
    # avoid zero range
    a_range = (a_max - a_min).clamp_min(1e-6)
    L_range = (L_max - L_min).clamp_min(1e-6)
    a_n = (a - a_min) / a_range
    L_n = (L - L_min) / L_range
    # discretize
    a_bin = torch.clamp((a_n * (bins - 1)).long(), 0, bins - 1)  # [N,H]
    L_bin = torch.clamp((L_n * (bins - 1)).long(), 0, bins - 1)
    mi = torch.zeros(H, dtype=torch.float32)
    # Iterate neurons (H could be large; keep loops in Python acceptable for moderate sizes)
    for i in range(H):
        xi = a_bin[:, i]
        yi = L_bin[:, i]
        # joint counts
        joint = torch.zeros(bins, bins, dtype=torch.float32)
        idx2 = xi * bins + yi
        # bincount
        flat_counts = torch.bincount(idx2, minlength=bins * bins).float().reshape(bins, bins)
        joint = flat_counts / float(N)
        px = joint.sum(dim=1, keepdim=True)
        py = joint.sum(dim=0, keepdim=True)
        # avoid zeros
        mask = joint > 0
        numer = joint[mask]
        denom = (px @ py)[mask]
        val = (numer * (numer / denom).log()).sum()
        if torch.isfinite(val):
            mi[i] = val
    # clamp negative drift
    mi = mi.clamp_min(0.0)
    return mi


def _compute_tfi(block: Dict[str, torch.Tensor], eps: float = EPS) -> Dict[str, torch.Tensor]:
    phi_a = block['phi_a']
    phi_g = block['phi_g']
    phi_L = block['phi_L']
    # Hess diag proxy
    h_diag = phi_g.pow(2) + phi_L
    block['h_diag'] = h_diag

    # Default MI surrogate = |phi_a|
    mi_vec = phi_a.abs()
    # If per-sample stats available, compute MI
    if 'phi_a_samples' in block and 'phi_L_samples' in block:
        try:
            A_s = block['phi_a_samples']  # [N,H]
            L_s = block['phi_L_samples']  # [N,H]
            if A_s.dim() == 2 and L_s.dim() == 2 and A_s.shape == L_s.shape and A_s.shape[0] >= 8:
                mi_vec = _estimate_mi_from_samples(A_s, L_s)
        except Exception:
            pass
    block['mi_phi_a_L'] = mi_vec
    tfi = phi_g.abs() * mi_vec / (h_diag + eps)
    block['tfi'] = tfi
    return block


def _cluster_neurons(block: Dict[str, torch.Tensor], K: int, random_state: int = 42) -> torch.Tensor:
    H = block['phi_a'].shape[0]
    if K <= 1 or H <= 1:
        return torch.zeros(H, dtype=torch.long)
    feats = torch.stack([
        block['phi_a'], block['phi_g'], block['phi_L'], block['tfi']
    ], dim=1).numpy()
    # 标准化
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    feats_n = (feats - mean) / std
    if _HAVE_SK:
        # 保证 K 不超过 H
        K_eff = min(K, H)
        if K_eff <= 1:
            return torch.zeros(H, dtype=torch.long)
        km = KMeans(n_clusters=K_eff, n_init='auto' if hasattr(KMeans, '__doc__') else 10, random_state=random_state)
        labels = km.fit_predict(feats_n)
        return torch.from_numpy(labels.astype(np.int64))
    # fallback: energy threshold -> 两簇 (高 tfi / 低 tfi)
    tfi = block['tfi'].numpy()
    med = float(np.median(tfi))
    labels = (tfi > med).astype(np.int64)
    return torch.from_numpy(labels)


def _aggregate_ensembles(block: Dict[str, torch.Tensor], labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (centroids [K,3], h_mean[K], counts[K])"""
    phi_a = block['phi_a']; phi_g = block['phi_g']; phi_L = block['phi_L']; h_diag = block['h_diag']
    K = int(labels.max().item()) + 1 if labels.numel() > 0 else 0
    if K == 0:
        return torch.empty(0,3), torch.empty(0), torch.empty(0, dtype=torch.long)
    centroids = torch.zeros(K, 3, dtype=torch.float32)
    h_mean = torch.zeros(K, dtype=torch.float32)
    counts = torch.zeros(K, dtype=torch.long)
    for k in range(K):
        idx = (labels == k).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        centroids[k, 0] = phi_a[idx].mean()
        centroids[k, 1] = phi_g[idx].mean()
        centroids[k, 2] = phi_L[idx].mean()
        h_mean[k] = h_diag[idx].mean()
        counts[k] = idx.numel()
    return centroids, h_mean, counts


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + eps
    return num / den


def _match_ensembles(centA: torch.Tensor, hA: torch.Tensor, centB: torch.Tensor, hB: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (pairs[Km,2], scores[Km]) with Hungarian/greedy."""
    KA = centA.shape[0]; KB = centB.shape[0]
    if KA == 0 or KB == 0:
        return torch.empty(0,2, dtype=torch.long), torch.empty(0), torch.empty(0)
    # compute cosine matrix
    # expand for broadcast: [KA,1,3] * [1,KB,3]
    a_exp = centA.unsqueeze(1)
    b_exp = centB.unsqueeze(0)
    num = (a_exp * b_exp).sum(dim=-1)
    den = a_exp.norm(dim=-1) * b_exp.norm(dim=-1) + EPS
    cos_mat = num / den
    # curvature penalty
    h_diff = (hA.unsqueeze(1) - hB.unsqueeze(0)).abs()
    penalty = torch.exp(-beta * h_diff)
    tos = cos_mat * penalty  # [KA, KB]
    tos_np = tos.numpy()

    if _HAVE_SCIPY:
        cost = -tos_np
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = []
        scores = []
        for r, c in zip(row_ind, col_ind):
            pairs.append((int(r), int(c)))
            scores.append(float(tos_np[r, c]))
    else:
        # greedy
        flat = np.dstack(np.unravel_index(np.argsort(tos_np, axis=None)[::-1], tos_np.shape))[0]
        usedA = set(); usedB = set(); pairs = []; scores = []
        for r, c in flat:
            if r in usedA or c in usedB:
                continue
            pairs.append((int(r), int(c)))
            scores.append(float(tos_np[r, c]))
            usedA.add(r); usedB.add(c)
            if len(usedA) >= KA or len(usedB) >= KB:
                break
    if not pairs:
        return torch.empty(0,2, dtype=torch.long), torch.empty(0), tos
    pairs_t = torch.tensor(pairs, dtype=torch.long)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    return pairs_t, scores_t, tos

# -----------------------------
# Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="TEFM Stage-2: 轨迹导向 ensemble 定位与映射 (LAPE)")
    ap.add_argument('--acts-a', type=str, required=True, help='模型A averaged activations .pt')
    ap.add_argument('--acts-a-lape', type=str, default=None, help='模型A LAPE sidecar (若不提供自动推断 *_lape.pt)')
    ap.add_argument('--acts-b', type=str, required=True, help='模型B averaged activations .pt')
    ap.add_argument('--acts-b-lape', type=str, default=None, help='模型B LAPE sidecar')
    ap.add_argument('--save', type=str, default=None, help='输出文件路径 (.pt)')
    ap.add_argument('--k', type=int, default=8, help='默认每个模块聚类簇数（两侧独立，若 H < k 自动降级）')
    ap.add_argument('--beta', type=float, default=0.1, help='曲率差异惩罚系数 beta (TOS 中 exp(-beta * |hA - hB|))')
    ap.add_argument('--min-neurons', type=int, default=8, help='若模块神经元数 < min-neurons 则跳过（避免噪声）')
    ap.add_argument('--tfi-top-ratio', type=float, default=None, help='可选：仅保留 TFI Top-ratio 的神经元再聚类 (0~1)')
    ap.add_argument('--verbose', action='store_true', help='打印详细日志')
    ap.add_argument('--deterministic', action='store_true', help='使用固定随机种子 (42) 以实现确定性 KMeans / 聚类结果')
    return ap.parse_args()


def main():
    args = parse_args()

    actsA, lapeA = _load_acts_and_lape(args.acts_a, args.acts_a_lape)
    actsB, lapeB = _load_acts_and_lape(args.acts_b, args.acts_b_lape)

    if args.deterministic:
        import random, numpy as _np
        torch.manual_seed(42)
        random.seed(42)
        _np.random.seed(42)

    inter_mods = sorted(set(actsA.keys()) & set(actsB.keys()) & set(lapeA.keys()) & set(lapeB.keys()))
    if args.verbose:
        print(f"[Info] 可对齐模块数: {len(inter_mods)}")

    modules_out: Dict[str, Any] = {}
    skipped = []
    for name in inter_mods:
        blkA = _prepare_module_block(actsA[name], lapeA.get(name))
        blkB = _prepare_module_block(actsB[name], lapeB.get(name))
        if blkA is None or blkB is None:
            skipped.append(name); continue
        H_A = blkA['phi_a'].shape[0]; H_B = blkB['phi_a'].shape[0]
        if H_A < args.min_neurons or H_B < args.min_neurons:
            skipped.append(name); continue

        # 计算 TFI / h_diag
        _compute_tfi(blkA); _compute_tfi(blkB)

        # 可选：基于 TFI 筛掉尾部神经元（保持索引对应逻辑，统一保留 -> 直接筛）
        if args.tfi_top_ratio is not None and 0 < args.tfi_top_ratio < 1.0:
            def _filter_by_tfi(block: Dict[str, torch.Tensor]):
                tfi = block['tfi']
                H = tfi.shape[0]
                k_keep = max(1, int(round(H * args.tfi_top_ratio)))
                idx = torch.topk(tfi, k_keep).indices
                for key in ['phi_a','phi_g','phi_L','tfi','h_diag','output','input','fai','fai_A','fai_MI','fai_H']:
                    if key in block:
                        block[key] = block[key][idx]
            _filter_by_tfi(blkA); _filter_by_tfi(blkB)
            H_A = blkA['phi_a'].shape[0]; H_B = blkB['phi_a'].shape[0]
            if H_A < 2 or H_B < 2:
                skipped.append(name); continue

        # 聚类（独立）
        K_base = max(1, int(args.k))
        labelsA = _cluster_neurons(blkA, K_base, random_state=42 if args.deterministic else 42)
        labelsB = _cluster_neurons(blkB, K_base, random_state=42 if args.deterministic else 42)
        centA, hA, countsA = _aggregate_ensembles(blkA, labelsA)
        centB, hB, countsB = _aggregate_ensembles(blkB, labelsB)

        # 匹配
        pairs, scores, tos_mat = _match_ensembles(centA, hA, centB, hB, beta=float(args.beta))

        modules_out[name] = {
            'ensembles_A': labelsA,
            'ensembles_B': labelsB,
            'K_A': centA.shape[0],
            'K_B': centB.shape[0],
            'centroids_A': centA,
            'centroids_B': centB,
            'h_mean_A': hA,
            'h_mean_B': hB,
            'pairs': pairs,
            'scores': scores,
            'module_size_A': H_A,
            'module_size_B': H_B,
            'tfi_A': blkA['tfi'],
            'tfi_B': blkB['tfi'],
            'tos_matrix': tos_mat,  # 便于调试/可视化
        }
        if args.verbose:
            print(f"[Module] {name}: H_A={H_A} H_B={H_B} K_A={centA.shape[0]} K_B={centB.shape[0]} mapped={pairs.shape[0]}")

    if args.verbose and skipped:
        print(f"[Info] 跳过模块: {len(skipped)} -> {skipped[:8]}{'...' if len(skipped)>8 else ''}")

    save_path = args.save
    if not save_path:
        os.makedirs('activations', exist_ok=True)
        labelA = osp.basename(args.acts_a).replace('.pt','')
        labelB = osp.basename(args.acts_b).replace('.pt','')
        save_path = osp.join('activations', f"tefm_stage2_{labelA}_TO_{labelB}.pt")

    torch.save({
        'meta': {
            'acts_a': args.acts_a,
            'acts_b': args.acts_b,
            'acts_a_lape': args.acts_a_lape,
            'acts_b_lape': args.acts_b_lape,
            'k': args.k,
            'beta': float(args.beta),
            'min_neurons': int(args.min_neurons),
            'tfi_top_ratio': args.tfi_top_ratio,
            'have_sklearn': _HAVE_SK,
            'have_scipy': _HAVE_SCIPY,
        },
        'modules': modules_out,
    }, save_path)

    print(f"[Done] TEFM Stage-2 ensemble & mapping saved to: {save_path}")


if __name__ == '__main__':
    main()
