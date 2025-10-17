# Stage-1 (base)
python -m merge.dgsm_v0_5_2.dgsm_stage1_subspace \
  --model-dir downloaded_models/mplug-owl2-llama2-7b \
  --rank 128 \
  --save work/dgsm_v0_5_2/stage1_A_r128_lm.pt \
  --cuda

# Stage-1 (donor)
python -m merge.dgsm_v0_5_2.dgsm_stage1_subspace \
  --model-dir downloaded_models/llava-v1.5-7b \
  --rank 128 \
  --save work/dgsm_v0_5_2/stage1_B_r128_lm.pt \
  --cuda

# Stage-2
python -m merge.dgsm_v0_5_2.dgsm_stage2_dynamic_gwd \
  --subs-a work/dgsm_v0_5_2/stage1_A_r128_lm.pt \
  --subs-b work/dgsm_v0_5_2/stage1_B_r128_lm.pt \
  --save work/dgsm_v0_5_2/stage2_r128_us_dynamic_lm.pt \
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
  --base-model downloaded_models/llava-v1.5-7b \
  --donor-model downloaded_models/mplug-owl2-llama2-7b \
  --stage2 work/dgsm_v0_5_2/stage2_r128_us_dynamic_lm.pt \
  --output-dir dgsm_merged_models_stage3_v0_5_2 \
  --base-subs work/dgsm_v0_5_2/stage1_A_r128_lm.pt \
  --use-dynamic-m \
  --use-lambda-est \
  --ortho-adapt \
  --tfi-beta 0.1 \
  --lambda-beta 0.1 \
  --tfi-threshold 0.01 \
  --tfi-topk 96 \
  --mapping-beta 0.1