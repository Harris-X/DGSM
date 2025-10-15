# Stage-1 (base)
python -m merge.dgsm_v0_5_2.dgsm_stage1_subspace \
  --model-dir downloaded_models/Qwen2-VL-7B-Instruct \
  --rank 128 \
  --save work/dgsm_v0_5_2/stage1_A_r128.pt \
  --cuda

# Stage-1 (donor)
python -m merge.dgsm_v0_5_2.dgsm_stage1_subspace \
  --model-dir downloaded_models/llava-onevision-qwen2-7b-si-hf \
  --rank 128 \
  --save work/dgsm_v0_5_2/stage1_B_r128.pt \
  --cuda

# Stage-2
python -m merge.dgsm_v0_5_2.dgsm_stage2_dynamic_gwd \
  --subs-a work/dgsm_v0_5_2/stage1_A_r128.pt \
  --subs-b work/dgsm_v0_5_2/stage1_B_r128.pt \
  --save work/dgsm_v0_5_2/stage2_r128_us_dynamic.pt \
  --dist-mode us \
  --dynamic-steps 15 \
  --dynamic-lr 5e-3 \
  --dynamic-reg 1e-3 \
  --gamma 4.5 \
  --cost-scale 48 \
  --use-pot --pot-method gw --pot-eps 0.5 \
  --dynamic-report

# Stage-3
python -m merge.dgsm_v0_5_2.dgsm_stage3_merge \
  --base-model downloaded_models/Qwen2-VL-7B-Instruct \
  --donor-model downloaded_models/llava-onevision-qwen2-7b-si-hf \
  --stage2 work/dgsm_v0_5_2/stage2_r128_us_dynamic.pt \
  --output-dir dgsm_merged_models_stage3_v0_5_2 \
  --base-subs work/dgsm_v0_5_2/stage1_A_r128.pt \
  --use-dynamic-m \
  --use-lambda-est \
  --ortho-adapt \
  --tfi-beta 0.1 \
  --lambda-beta 0.1 \
  --tfi-threshold 0.02 \
  --tfi-topk 96 \
  --mapping-beta 0.1