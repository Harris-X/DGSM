# Stage-1: 元数据（任务向量统计）
python merge/ATAF/ataf_stage1_taskvec.py \
  --model-a downloaded_models/mplug-owl2-llama2-7b \
  --model-b downloaded_models/llava-v1.5-7b \
  --base-model downloaded_models/Llama-2-7b-hf \
  --save work/outputs_ataf/stage1_meta.pt

# Stage-2: 列空间 Procrustes + 列缩放对齐
python merge/ATAF/ataf_stage2_align.py \
  --stage1 work/outputs_ataf/stage1_meta.pt \
  --model-a downloaded_models/mplug-owl2-llama2-7b \
  --model-b downloaded_models/llava-v1.5-7b \
  --base-model downloaded_models/Llama-2-7b-hf \
  --save work/outputs_ataf/stage2_align.pt \
  --max-cols 4096

# Stage-3: 自适应 α 融合（锚定 A） #   --adaptive-alpha \
python merge/ATAF/ataf_stage3_fuse.py \
  --stage2 work/outputs_ataf/stage2_align.pt \
  --model-a downloaded_models/mplug-owl2-llama2-7b \
  --model-b downloaded_models/llava-v1.5-7b \
  --base-model downloaded_models/Llama-2-7b-hf \
  --output-dir work/outputs_ataf \
  --alpha 0.2 \
  --bias-alpha 0.2 \
  --alpha-max 0.8 \
  --gain-scale 4.0