import torch
pkg = torch.load("/root/autodl-tmp/AdaMMS/work/dgsm/stage2_r128_us_on_gw_e0.5_it30_reg0.05_ds8.pt", map_location="cpu") # mPLUG-Owl2_meta llava_v1.5_7b
# acts = pkg["activations"]
# meta = pkg["meta"]
# print(pkg["layers.2"]["input"].shape)
print(pkg)
# first_key = sorted(acts.keys())[0]
# print(first_key, acts[first_key].keys(), acts[first_key].get("output").shape)