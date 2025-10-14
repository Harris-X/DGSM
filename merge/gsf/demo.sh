# Stage-1: 提取子空间 (base)
python merge/gsf/gsf_stage1_subspace.py \
  --model-dir downloaded_models/mplug-owl2-llama2-7b \
  --rank 64 \
  --save activations/gsf_stage1_mplug.pt --verbose

# Stage-1: 提取子空间 (donor)
python merge/gsf/gsf_stage1_subspace.py \
  --model-dir downloaded_models/llava-v1.5-7b \
  --rank 64 \
  --save activations/gsf_stage1_llava.pt --verbose

# Stage-2: GWD 对齐 (+ POT)
python merge/gsf/gsf_stage2_gwd.py \
  --subs-a activations/mplug_subspace.pt \
  --subs-b activations/llava_subspace.pt \
  --save activations/gsf_stage2_mplug_TO_llava.pt \
  --dist-mode us --use-pot \
  --iters 30 --sink-reg 0.05 --tol 5e-4 --patience 3 \
  --gamma 4.0 --cost-scale 1.0 --verbose

# Stage-2b: 行级 / 组级 λ
python merge/gsf/gsf_stage2b_group_loc.py \
  --subs-a activations/mplug_subspace.pt  \
  --subs-b activations/llava_subspace.pt \
  --stage2 activations/gsf_stage2_mplug_TO_llava.pt \
  --save activations/gsf_stage2b_mplug_TO_llava.pt \
  --k 8 --gamma 4.0 --cost-scale 1.0 --alpha 0.5 --beta 0.5 --verbose

# Stage-3: 融合
python merge/gsf/gsf_stage3_merge.py \
  --base-model downloaded_models/mplug-owl2-llama2-7b \
  --donor-model downloaded_models/llava-v1.5-7b \
  --stage2 activations/gsf_stage2_mplug_TO_llava.pt \
  --stage2b activations/gsf_stage2b_mplug_TO_llava.pt \
  --output-dir merged_models_stage3 \
  --cost-scale 1.0 --gamma 4.0 --ortho-scale 0.5 \
  --fallback-alpha 0.5 --bias-alpha 0.5

# 一键全流程（若使用示例脚本）
bash run/gsf_pipeline_example.sh