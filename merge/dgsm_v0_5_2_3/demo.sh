# Stage-1 (base)
python -m merge.dgsm_v0_5_2_3.dgsm_stage1_subspace \
  --model-dir downloaded_models/Qwen2-VL-7B-Instruct \
  --save work/dgsm_v0_5_2_3/stage1_A_r128.pt \
  --rank 128 \
  --cuda


# Stage-1 (donor)
python -m merge.dgsm_v0_5_2_3.dgsm_stage1_subspace \
  --model-dir downloaded_models/llava-onevision-qwen2-7b-si-hf \
  --save work/dgsm_v0_5_2_3/stage1_B_r128.pt \
  --rank 128 \
  --cuda


# Stage-2
python -m merge.dgsm_v0_5_2_3.dgsm_stage2_dynamic_gwd \
  --subs-a work/dgsm_v0_5_2_3/stage1_A_r128.pt \
  --subs-b work/dgsm_v0_5_2_3/stage1_B_r128.pt \
  --save work/dgsm_v0_5_2_3/stage2_r128_dyn.pt \
  --dist-mode us \
  --dynamic-steps 30 \
  --dynamic-lr 5e-3 \
  --dynamic-reg 1e-3 \
  --dyn-loss hybrid \
  --dyn-mix-alpha 0.2 \
  --entropic-eps 0.1 \
  --entropic-iters 5 \
  --entropic-sinkhorn 5 \
  --iters 30 \
  --sink-reg 0.05 \
  --max-rank 128 \
  --verbose


# Stage-3
python -m merge.dgsm_v0_5_2_3.dgsm_stage3_merge \
  --base-model downloaded_models/Qwen2-VL-7B-Instruct \
  --donor-model downloaded_models/llava-onevision-qwen2-7b-si-hf \
  --stage2 work/dgsm_v0_5_2/stage2_r128_us_dynamic.pt \
  --output-dir dgsm_merged_models_stage3_v0_5_2_3 \
  --base-subs work/dgsm_v0_5_2/stage1_A_r128.pt \
  --use-dynamic-m \
  --use-lambda-est \
  --ortho-adapt \
  --tfi-beta 0.1 \
  --lambda-beta 0.1 \
  --tfi-threshold 0.01 \
  --tfi-topk 96 \
  --mapping-beta 0.1 \
  --mapping-cosine-mode psi_prime 

python -m merge.dgsm_v0_5_2_3.dgsm_stage3_merge \
  --base-model downloaded_models/Qwen2-VL-7B-Instruct \
  --donor-model downloaded_models/llava-onevision-qwen2-7b-si-hf \
  --stage2 work/dgsm_v0_5_2/stage2_r128_us_dynamic.pt \
  --output-dir dgsm_merged_models_stage3_v0_5_2_3 \
  --base-subs work/dgsm_v0_5_2/stage1_A_r128.pt \
  --use-dynamic-m \
  --use-lambda-est \
  --ortho-adapt \
  --tfi-beta 0.1 \
  --lambda-beta 0.1 \
  --tfi-threshold 0.01 \
  --tfi-topk 96 \
  --mapping-beta 0.1 \
  --mapping-cosine-mode psi 