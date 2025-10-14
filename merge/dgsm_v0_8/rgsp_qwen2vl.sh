#!/usr/bin/env bash
set -euo pipefail

# RGSP-TEFM pipeline for Qwen2-VL ↔️ LLaVA-OneVision-Qwen2
# Generates RGSP Stage-1/2/3 artifacts under activations/ and merged_models_stage3/

BASE_MODEL="downloaded_models/Qwen2-VL-7B-Instruct"
DONOR_MODEL="downloaded_models/llava-onevision-qwen2-7b-si-hf"
RANK=128

STAGE1_BASE="activations/rgsp_qwen2vl_stage1_base.pt"
STAGE1_DONOR="activations/rgsp_qwen2vl_stage1_donor.pt"
STAGE2_OUT="activations/rgsp_qwen2vl_stage2_base_TO_donor.pt"
MERGE_OUT_DIR="merged_models_stage3"

mkdir -p "${MERGE_OUT_DIR}" activations

echo "[RGSP][Stage-1] Extracting subspaces..."
python merge/dgsm_v0_8/dgsm_stage1_subspace.py \
  --model-dir "${BASE_MODEL}" \
  --rank "${RANK}" \
  --save "${STAGE1_BASE}" \
  --cuda

python merge/dgsm_v0_8/dgsm_stage1_subspace.py \
  --model-dir "${DONOR_MODEL}" \
  --rank "${RANK}" \
  --save "${STAGE1_DONOR}" \
  --cuda

echo "[RGSP][Stage-2] Running dynamic entropic GWD alignment..."
python merge/dgsm_v0_8/dgsm_stage2_dynamic_gwd.py \
  --subs-a "${STAGE1_BASE}" \
  --subs-b "${STAGE1_DONOR}" \
  --save "${STAGE2_OUT}" \
  --dist-mode us \
  --gamma 4.0 \
  --cost-scale 32.0 \
  --dynamic-steps 10 \
  --dynamic-lr 1e-3 \
  --dynamic-reg 5e-4 \
  --dyn-loss entropic \
  --entropic-eps 0.1 \
  --entropic-iters 8 \
  --entropic-sinkhorn 12 \
  --use-pot \
  --pot-method entropic \
  --pot-eps 0.35 \
  --pot-max-iter 80 \
  --verbose

echo "[RGSP][Stage-3] Fusing with relative subspace projection..."
python merge/dgsm_v0_8/dgsm_stage3_merge.py \
  --base-model "${BASE_MODEL}" \
  --donor-model "${DONOR_MODEL}" \
  --stage2 "${STAGE2_OUT}" \
  --output-dir "${MERGE_OUT_DIR}" \
  --use-dynamic-m \
  --base-subs "${STAGE1_BASE}" \
  --donor-subs "${STAGE1_DONOR}" \
  --fallback-alpha 0.5 \
  --bias-alpha 0.5 \
  --tfi-threshold 0.0 \
  --tfi-topk 16 \
  --mapping-beta 0.12

echo "[RGSP] Pipeline complete. Merged weights saved under ${MERGE_OUT_DIR}/$(basename "${BASE_MODEL}")/dgsm_merged"
