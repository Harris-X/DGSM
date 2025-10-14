import torch
pkg = torch.load("activations/gsf_stage2_mplug_TO_llava.pt", map_location="cpu")
# acts = pkg["activations"]
# meta = pkg["meta"]
# print(pkg["layers.2"]["input"].shape)
print(pkg)
# first_key = sorted(acts.keys())[0]
# print(first_key, acts[first_key].keys(), acts[first_key].get("output").shape)