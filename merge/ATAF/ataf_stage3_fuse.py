#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATAF Stage-3: Aligned Task Vector Fusion (ANCHOR = Model A)

根据用户最新需求：最终融合结果应“往多模态大模型 A 合并”，输出模型保持 A 的整体结构与除被融合线性层外的其它模块（视觉编码器 / 特定头等）完全一致，Base(C) 仅作为共同祖先用于定义任务向量，不再作为最终权重锚点。

读取 Stage-2 生成的对齐信息 (每个 2D 线性层的 P 与 λ) 后，对每个需要融合的 2D 权重执行：
    tau_A = W_A - W_C
    tau_B = W_B - W_C
    对 B 的任务向量做列截断 + 旋转 + 列缩放得到 tau_B_full (若列被截断则补回未截断部分原始差分)；
    计算增量 extra = tau_B_full - tau_A  (表示相对 A 还缺失 / 需要注入的新增知识方向)。
    融合后权重： W_fused = W_A + alpha * extra

性质：
    * alpha=0 -> 原样保留 A
    * alpha=1 -> 等价于 W_C + tau_B_full （即完全“迁移”到对齐后的 B 差分位置）
    * 不直接回落到 Base; 仅利用 Base 统一坐标系，避免直接 A/B 权重差异畸形叠加。

Bias：由于 bias 的任务向量差分 extra = (b_B - b_C) - (b_A - b_C) = b_B - b_A，因此锚定 A 下融合等价于线性插值：b_fused = (1-alpha_b)*b_A + alpha_b*b_B。

Adaptive Alpha（可选）：与旧实现一致，仅改变其作用是在“增量”尺度上调节注入强度。

保存：复制 model_a 目录结构（而非 base_model）到 <output_dir>/<basename(model_a)>/ataf_fused。
"""
from __future__ import annotations
import argparse, json, math, os, os.path as osp, datetime
from collections import defaultdict
from typing import Dict
import torch, safetensors.torch
from tqdm import tqdm
from utils import load_weights, need_merge  # type: ignore

EPS=1e-8

def _sigmoid(x: float) -> float:
    try:
        return 1.0/(1.0+math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _save_model(args, merged: Dict[str, torch.Tensor]):
    # 以 model_a 为结构模板保存（符合“最终权重也是多模态大模型A”要求）
    base_dir = osp.basename(args.model_a.rstrip(os.sep))
    out_root = osp.join(args.output_dir, base_dir, 'ataf_fused')
    os.makedirs(out_root, exist_ok=True)
    # 复用 A 模型 shard 结构
    sft_index = osp.join(args.model_a, 'model.safetensors.index.json')
    bin_index = osp.join(args.model_a, 'pytorch_model.bin.index.json')

    def copy_side():
        for fn in os.listdir(args.model_a):
            if fn.endswith(('.json', '.model', '.py', '.md')):
                src = osp.join(args.model_a, fn); dst = osp.join(out_root, fn)
                if not osp.exists(dst):
                    try:
                        import shutil; shutil.copy(src, dst)
                    except Exception:
                        pass

    if osp.exists(sft_index):
        with open(sft_index, 'r') as f: index = json.load(f)['weight_map']
        shards=defaultdict(dict)
        for k,v in merged.items():
            if k in index: shards[index[k]][k]=v
        for shard, sd in shards.items(): safetensors.torch.save_file(sd, osp.join(out_root, shard))
        copy_side(); print(f"[Save] Sharded safetensors -> {out_root}"); return out_root
    if osp.exists(bin_index):
        with open(bin_index, 'r') as f: index = json.load(f)['weight_map']
        shards=defaultdict(dict)
        for k,v in merged.items():
            if k in index: shards[index[k]][k]=v
        for shard, sd in shards.items(): torch.save(sd, osp.join(out_root, shard))
        copy_side(); print(f"[Save] Sharded .bin -> {out_root}"); return out_root
    sft_single=osp.join(args.model_a,'model.safetensors')
    bin_single=osp.join(args.model_a,'pytorch_model.bin')
    if osp.exists(sft_single):
        safetensors.torch.save_file(merged, osp.join(out_root,'model.safetensors')); copy_side(); print(f"[Save] Single safetensors -> {out_root}"); return out_root
    if osp.exists(bin_single):
        torch.save(merged, osp.join(out_root,'pytorch_model.bin')); copy_side(); print(f"[Save] Single .bin -> {out_root}"); return out_root
    safetensors.torch.save_file(merged, osp.join(out_root,'model.safetensors')); copy_side(); print(f"[Save] Default safetensors -> {out_root}"); return out_root


def ataf_fuse(args: argparse.Namespace):
    print("\n--- [ATAF Stage-3: Fusion] ---")
    stage2 = torch.load(args.stage2, map_location='cpu')
    align_info = stage2['align']
    weights_A = load_weights(args.model_a)
    weights_B = load_weights(args.model_b)
    weights_C = load_weights(args.base_model)

    # 起始于 A 的完整权重副本，保持多模态结构（视觉/投影等）
    merged = weights_A.copy()
    stat_linear = stat_used = 0

    for k, WA in tqdm(weights_A.items(), desc='ATAF Fuse'):
        if k not in weights_B or k not in weights_C:
            continue
        if not need_merge(k):
            continue
        WB = weights_B[k]
        WC = weights_C[k]
        if WA.shape != WB.shape or WA.shape != WC.shape:
            continue
        if WA.ndim == 2:  # 线性层
            stat_linear += 1
            tauA = WA.float() - WC.float()
            tauB = WB.float() - WC.float()
            rec = align_info.get(k)
            if rec is not None:
                r_cols = rec['rank_cols']
                P = rec['P'].float()  # r_cols x r_cols
                lam = rec['lambda'].float()  # r_cols
                d_out, d_in = WA.shape
                r_cols = min(r_cols, d_in)
                tauB_use = tauB[:, :r_cols]
                tauB_rot = tauB_use @ P
                tauB_al = tauB_rot * lam.unsqueeze(0)
                if d_in > r_cols:
                    tauB_full = torch.cat([tauB_al, tauB[:, r_cols:]], dim=1)
                else:
                    tauB_full = tauB_al
                # Adaptive alpha (optional)
                alpha = args.alpha
                if args.adaptive_alpha and rec.get('pre_cos') is not None and rec.get('post_cos') is not None:
                    gain = rec['post_cos'] - (rec['pre_cos'] if rec['pre_cos'] is not None else 0.0)
                    alpha_gain = _sigmoid(args.gain_scale * gain)
                    alpha = min(args.alpha_max, alpha * alpha_gain)
                # 以 A 为锚：extra = tauB_full - tauA; W_new = W_A + alpha * extra
                extra = tauB_full - tauA
                W_new = WA.float() + alpha * extra
                merged[k] = W_new.to(WA.dtype)
                stat_used += 1
            else:
                # 无对齐信息：锚定 A 的简单插值 (A + alpha*(B-A))
                merged[k] = (WA.float() + args.alpha * (WB.float() - WA.float())).to(WA.dtype)
        elif WA.ndim == 1:  # bias
            bA = WA.float(); bB = WB.float()
            merged[k] = ((1 - args.bias_alpha) * bA + args.bias_alpha * bB).to(WA.dtype)

    out_root = _save_model(args, merged)
    meta = dict(
        model_a=args.model_a,
        model_b=args.model_b,
        base_model=args.base_model,
        stage2=args.stage2,
        alpha=args.alpha,
        adaptive_alpha=bool(args.adaptive_alpha),
        alpha_max=args.alpha_max,
        gain_scale=args.gain_scale,
        bias_alpha=args.bias_alpha,
        stat_linear=stat_linear,
        stat_used=stat_used,
        datetime=str(datetime.datetime.now()),
        output_dir=out_root,
        anchor='model_a'
    )
    with open(osp.join(out_root,'ataf_meta.json'),'w') as f: json.dump(meta,f,indent=2)
    print(f"[Done] ATAF fusion complete: linear={stat_linear}, used_align={stat_used}, output={out_root}")


def parse_args():
    ap = argparse.ArgumentParser(description='ATAF Stage-3 Fusion')
    ap.add_argument('--stage2', required=True)
    ap.add_argument('--model-a', required=True)
    ap.add_argument('--model-b', required=True)
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--bias-alpha', type=float, default=0.5)
    ap.add_argument('--adaptive-alpha', action='store_true')
    ap.add_argument('--alpha-max', type=float, default=0.9)
    ap.add_argument('--gain-scale', type=float, default=4.0)
    return ap.parse_args()

if __name__ == '__main__':
    ataf_fuse(parse_args())
