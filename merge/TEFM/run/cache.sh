python merge/TEFM/cache_activation_new.py \
  --gpus 1,2,3,6 \
  --model llava_v1.5_7b \
  --hf-dataset meta --hf-offline \
  --req-act input output \
  --module-regex "mlp\.|self_attn\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn" \
  --lape-enable --lape-samples 2 --lape-gamma 0.99 \
  --lape-top-p 0.9 --lape-temperature 0.7 \
  --lape-min-new 1 --lape-max-new 4 \
  --max-samples 8

python merge/TEFM/cache_activation_new.py \
  --gpus 0,1,2,3 \
  --model mPLUG-Owl2 \
  --hf-dataset meta --hf-offline \
  --req-act input output \
  --module-regex "mlp\.|self_attn\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn" \
  --lape-enable --lape-samples 2 --lape-gamma 0.99 \
  --lape-top-p 0.9 --lape-temperature 0.7 \
  --lape-min-new 1 --lape-max-new 4 \
  --max-samples 8

python merge/TEFM/cache_activation_new.py \
  --gpus 0,1,2,3 \
  --model mPLUG-Owl2 \
  --hf-dataset meta --hf-offline \
  --req-act input output \
  --module-regex "mlp\\.|self_attn\\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn" \
  --lape-enable --lape-samples 2 --lape-gamma 0.99 \
  --lape-top-p 0.9 --lape-temperature 0.7 \
  --lape-min-new 1 --lape-max-new 4 \
  --vlm-device-map auto \
  --max-samples 8 \
  --lape-mi-store --lape-mi-max-samples 512