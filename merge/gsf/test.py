import torch
pkg = torch.load("/root/autodl-tmp/AdaMMS/activations/dgsm_stage1_base.pt", map_location="cpu")
# acts = pkg["activations"]
# meta = pkg["meta"]
# print(pkg["layers.2"]["input"].shape)
print(pkg)
# first_key = sorted(acts.keys())[0]
# print(first_key, acts[first_key].keys(), acts[first_key].get("output").shape)