# Stage-1：分别抽取 base / donor 子空间
python merge/dgsm_v0_8/dgsm_stage1_subspace.py \
  --model-dir downloaded_models/mplug-owl2-llama2-7b \
  --rank 128 \
  --save activations/rgsp_stage1_base.pt \
  --cuda

python merge/dgsm_v0_8/dgsm_stage1_subspace.py \
  --model-dir downloaded_models/llava-v1.5-7b \
  --rank 128 \
  --save activations/rgsp_stage1_donor.pt \
  --cuda

# Stage-2：启用动态映射 + entropic GW，生成RGSP所需统计
python merge/dgsm_v0_8/dgsm_stage2_dynamic_gwd.py \
  --subs-a activations/rgsp_stage1_base.pt \
  --subs-b activations/rgsp_stage1_donor.pt \
  --save activations/rgsp_stage2_base_TO_donor.pt \
  --dist-mode us \
  --gamma 4.0 \
  --cost-scale 32.0 \
  --dynamic-steps 8 \
  --dynamic-lr 1e-3 \
  --dynamic-reg 5e-4 \
  --dynamic-gw-mode entropic \
  --dynamic-gw-eps 0.06 \
  --dynamic-gw-outer 3 \
  --dynamic-gw-sinkhorn 15 \
  --use-pot \
  --pot-method entropic \
  --pot-eps 0.05 \
  --verbose

# Stage-3：RGSP融合，需传入 base/donor Stage-1 结果并可选动态映射
python merge/dgsm_v0_8/dgsm_stage3_merge.py \
  --base-model downloaded_models/mplug-owl2-llama2-7b \
  --donor-model downloaded_models/llava-v1.5-7b \
  --stage2 activations/rgsp_stage2_base_TO_donor.pt \
  --output-dir merged_models_stage3 \
  --use-dynamic-m \
  --base-subs activations/rgsp_stage1_base.pt \
  --donor-subs activations/rgsp_stage1_donor.pt \
  --fallback-alpha 0.5 \
  --bias-alpha 0.5 \
  --tfi-threshold 0.0 \
  --tfi-topk 16 \
  --mapping-beta 0.12