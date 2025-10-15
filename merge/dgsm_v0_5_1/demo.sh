python merge/dgsm/dgsm_stage1_subspace.py --model-dir downloaded_models/Qwen2-VL-7B-Instruct --rank 128 --save activations/dgsm_stage1_base_qwen.pt
python merge/dgsm/dgsm_stage1_subspace.py --model-dir downloaded_models/llava-onevision-qwen2-7b-si-hf --rank 128 --save activations/dgsm_stage1_donor_qwen.pt

python merge/dgsm/dgsm_stage2_dynamic_gwd.py \
  --subs-a activations/dgsm_stage1_base_r128.pt \
  --subs-b activations/dgsm_stage1_donor_r128.pt \
  --save activations/dgsm_stage2_base_TO_donor_balanced.pt \
  --dist-mode us \
  --dynamic-steps 15 \
  --dyn-loss entropic \
  --entropic-eps 0.07 \
  --entropic-iters 4 \
  --entropic-sinkhorn 5 \
  --dynamic-lr 4e-3 \
  --dynamic-reg 7e-4 \
  --gamma 4.5 --cost-scale 48 \
  --max-rank 128 \
  --use-pot --pot-entropic-eps 0.04 \
  --dynamic-report --verbose

cd /root/autodl-tmp/AdaMMS && python -m merge.dgsm_v0_5.dgsm_stage3_merge --base-model downloaded_models/Qwen2-VL-7B-Instruct --donor-model downloaded_models/llava-onevision-qwen2-7b-si-hf --stage2 work/dgsm/stage2_r128_us_on_gw_e0.5_it30_reg0.05_ds8.pt --output-dir merged_models_stage3 --gamma 3.5 --cost-scale 0.8 --ortho-scale 0.3 --fallback-alpha 0.6 --bias-alpha 0.3 --use-dynamic-m --use-lambda-est --ortho-adapt --base-subs work/dgsm/stage1_A_r128.pt --tfi-threshold 0.1 --tfi-topk 96 --mapping-beta 0.32