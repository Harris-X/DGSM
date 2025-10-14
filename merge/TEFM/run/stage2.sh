python merge/TEFM/tefm_stage2_ensemble.py \
  --acts-a activations/mPLUG-Owl2_meta.pt \
  --acts-a-lape activations/mPLUG-Owl2_meta_lape.pt \
  --acts-b activations/llava_v1.5_7b_meta.pt \
  --acts-b-lape activations/llava_v1.5_7b_meta_lape.pt \
  --k 8 --beta 0.1 --verbose