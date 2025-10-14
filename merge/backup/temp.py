import os
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# --- 1. 核心配置 ---
CKPT_PATH = {
    "llama2": "./downloaded_models/Llama-2-7b-hf",
    "qwen2": "./downloaded_models/Qwen2-7B-Instruct",
}
# 定义计算设备和模型加载设备。如果显存不足，可将一个模型加载到CPU
MODEL_DEVICE_A = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
MODEL_DEVICE_B = torch.device("cuda:5" if torch.cuda.is_available() else "cpu") # 也可以是 "cpu"
COMPUTE_DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"模型A设备: {MODEL_DEVICE_A}, 模型B设备: {MODEL_DEVICE_B}, 计算设备: {COMPUTE_DEVICE}")

# --- 2. 辅助函数 (模型与数据加载) ---

def load_complete_model(model_id, device):
    """通用模型加载函数"""
    print(f"正在加载模型: {model_id} -> {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"{model_id.split('/')[-1]} 模型加载完成。")
    return tokenizer, model

def load_and_prepare_dataset(tokenizer_a, tokenizer_b, dataset_name="wikitext", split="test", max_samples=64, max_length=256):
    """加载并处理数据集以进行特征提取"""
    print(f"正在加载数据集: {dataset_name}...")
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    def tokenize_fn(examples):
        text = [t for t in examples["text"] if t and t.strip()]
        if not text: return {}
        inputs_a = tokenizer_a(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        inputs_b = tokenizer_b(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids_a": inputs_a.input_ids, "attention_mask_a": inputs_a.attention_mask,
            "input_ids_b": inputs_b.input_ids, "attention_mask_b": inputs_b.attention_mask,
        }

    processed_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    processed_dataset.set_format(type='torch')
    return processed_dataset

def get_module_by_name(model, module_name):
    """通过字符串名称安全地获取模块"""
    for part in module_name.split('.'):
        if not hasattr(model, part): return None
        model = getattr(model, part)
    return model

def register_hooks_for_reps(model, layer_names):
    """为指定层注册钩子以捕获输出激活，并立即转移到CPU"""
    reps, hooks = {name: [] for name in layer_names}, []
    hook_fn = lambda name: (lambda module, input, output: reps[name].append((output[0] if isinstance(output, tuple) else output).detach().cpu()))
    for name in layer_names:
        if module := get_module_by_name(model, name):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    return reps, hooks


# --- 3. 核心算法：CKA & 深度对齐 (LMA) ---

def cka(gram_k, gram_l):
    """计算中心核对齐(CKA)"""
    gram_k = center_gram(gram_k.float())
    gram_l = center_gram(gram_l.float())
    scaled_hsic = torch.sum(gram_k * gram_l)
    norm_k = torch.norm(gram_k)
    norm_l = torch.norm(gram_l)
    return scaled_hsic / (norm_k * norm_l) if norm_k != 0 and norm_l != 0 else torch.tensor(0.0)

def center_gram(gram):
    n = gram.shape[0]
    I = torch.eye(n, device=gram.device)
    H = I - 1/n * torch.ones(n, n, device=gram.device)
    return H @ gram @ H

def compute_cka_matrix(reps1, reps2, names1, names2, max_tokens=4096):
    """高效计算CKA相似度矩阵"""
    print("开始计算CKA矩阵...")
    cka_matrix = torch.zeros(len(names1), len(names2))
    
    processed_reps1 = {name: torch.cat(reps1[name], dim=0).flatten(0, 1).to(torch.float32) for name in names1}
    processed_reps2 = {name: torch.cat(reps2[name], dim=0).flatten(0, 1).to(torch.float32) for name in names2}

    for i, name1 in enumerate(tqdm(names1, desc="Deep Model Layers")):
        feat1_full = processed_reps1[name1]
        feat1 = feat1_full[torch.randperm(feat1_full.shape[0])[:max_tokens]].to(COMPUTE_DEVICE)
        gram_k = feat1 @ feat1.T
        
        for j, name2 in enumerate(names2):
            feat2_full = processed_reps2[name2]
            feat2 = feat2_full[torch.randperm(feat2_full.shape[0])[:max_tokens]].to(COMPUTE_DEVICE)
            gram_l = feat2 @ feat2.T
            
            min_dim = min(gram_k.shape[0], gram_l.shape[0])
            cka_matrix[i, j] = cka(gram_k[:min_dim, :min_dim], gram_l[:min_dim, :min_dim])
            
        del gram_k
        gc.collect()
        torch.cuda.empty_cache()

    print("CKA矩阵计算完成。")
    return cka_matrix.cpu()

def align_layers_lma(C):
    """使用LMA算法进行深度对齐"""
    m, n = C.shape
    F = torch.full((n + 1, m + 1), -torch.inf)
    F[0, :] = 0
    path = torch.zeros((n + 1, m + 1), dtype=torch.long)
    
    for i in range(1, n + 1):
        for j in range(i, m + 1):
            max_val, best_k = -torch.inf, -1
            for k in range(i - 1, j):
                segment_sim = C[k:j, i - 1].sum()
                current_val = F[i - 1, k] + segment_sim
                if current_val > max_val:
                    max_val = current_val
                    best_k = k
            F[i, j] = max_val
            path[i, j] = best_k

    alignment, i, j = [], n, m
    while i > 0:
        k = path[i, j]
        alignment.insert(0, list(range(k, j)))
        j = k
        i -= 1
        
    return alignment


# --- 4. 宽度异构合并：弹性神经元压缩 (ZipIt!) - CPU优化版 ---
def match_tensors_zipit(reps_a, reps_b, target_dim):
    """在CPU上执行ZipIt!算法，返回适用于两个模型的变换矩阵"""
    dim_a, dim_b = reps_a.shape[1], reps_b.shape[1]
    
    reps_a_cpu = reps_a.cpu().to(torch.float32)
    reps_b_cpu = reps_b.cpu().to(torch.float32)
    
    reps_a_norm = reps_a_cpu / (torch.norm(reps_a_cpu, p=2, dim=0, keepdim=True) + 1e-6)
    reps_b_norm = reps_b_cpu / (torch.norm(reps_b_cpu, p=2, dim=0, keepdim=True) + 1e-6)
    
    all_reps = torch.cat([reps_a_norm, reps_b_norm], dim=1)
    sim_matrix = all_reps.T @ all_reps
    sim_matrix.fill_diagonal_(-torch.inf)
    
    total_dim = dim_a + dim_b
    perm_matrix = torch.eye(total_dim, device='cpu', dtype=torch.bfloat16)

    num_merges = total_dim - target_dim
    for _ in tqdm(range(num_merges), desc=f"Zipping to {target_dim} dims on CPU", leave=False):
        flat_idx = torch.argmax(sim_matrix)
        idx1, idx2 = np.unravel_index(flat_idx.item(), sim_matrix.shape)
        
        if sim_matrix[idx1, idx2] == -torch.inf: break

        perm_matrix[:, idx1] += perm_matrix[:, idx2]
        sim_matrix[:, idx1] = (sim_matrix[:, idx1] + sim_matrix[:, idx2]) / 2
        sim_matrix[idx1, :] = (sim_matrix[idx1, :] + sim_matrix[idx2, :]) / 2
        
        perm_matrix[:, idx2] = 0
        sim_matrix[idx2, :] = -torch.inf
        sim_matrix[:, idx2] = -torch.inf
        sim_matrix[idx1, idx1] = -torch.inf

    unmerge_matrix = perm_matrix[:, perm_matrix.sum(dim=0) != 0]
    merge_matrix = unmerge_matrix.T
    merge_matrix = merge_matrix / (torch.sum(merge_matrix, dim=1, keepdim=True) + 1e-6)

    Tm_a, Tm_b = merge_matrix[:, :dim_a], merge_matrix[:, dim_a:]
    Tu_a, Tu_b = unmerge_matrix[:dim_a, :], unmerge_matrix[dim_a:, :]

    return Tm_a, Tm_b, Tu_a, Tu_b

def apply_hetero_merge_to_layer_final(base_layer, donor_layer, Tm_b, Tm_d, Tu_b, Tu_d, alpha):
    """
    最终修复版 v3: 严格遵循 W' = T_out @ W @ T_in⁻¹ 的双向变换原则。
    该版本修复了先前版本中矩阵乘法的维度错误。
    """
    dtype = base_layer.q_proj.weight.dtype
    device = base_layer.q_proj.weight.device

    # 将变换矩阵移动到计算设备并设置正确的类型
    Tm_b, Tm_d = Tm_b.to(device, dtype=dtype), Tm_d.to(device, dtype=dtype)
    Tu_b, Tu_d = Tu_b.to(device, dtype=dtype), Tu_d.to(device, dtype=dtype)

    # 1. 定义空间变换矩阵
    # 从 Donor 空间到 Base 空间的变换 T_d_to_b
    # 路径: Donor_Space -> Shared_Space -> Base_Space
    # T(Shared -> Base) = Tm_b (维度: d_base x d_shared)
    # T(Donor -> Shared) = Tu_d (维度: d_donor x d_shared)
    # 变换 donor 的输出: Y_b = Y_d @ Tu_d @ Tm_b  <- 这里之前的理解是错误的
    # 正确的应该是：从共享空间映射回来
    # T(Shared -> Base) 是 Tu_b (d_base, d_shared)
    # T(Donor -> Shared) 是 Tm_d (d_shared, d_donor)
    # 所以，T(Donor -> Base) = Tu_b @ Tm_d (d_base, d_donor)
    T_d_to_b = Tu_b @ Tm_d
    T_b_to_d = Tu_d @ Tm_b

    # 2. 变换 Q, K, V 投影层 (W: d_out, d_in)
    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
        base_proj = getattr(base_layer, proj_name)
        donor_proj = getattr(donor_layer, proj_name)

        # donor 权重 W_d 的形状是 (d_donor, d_donor)
        # base  权重 W_b 的形状是 (d_base, d_base)
        # 变换公式: W'_d = T_out @ W_d @ T_in⁻¹
        # 在这里，输入和输出空间都是模型隐藏层空间
        # T_out = T_d_to_b (d_base, d_donor)
        # T_in⁻¹ = T_d_to_b (因为 donor -> base), T_in = T_b_to_d, T_in⁻¹ = T_b_to_d⁻¹ ~= T_d_to_b
        W_d_transformed = T_d_to_b @ donor_proj.weight.data 
        W_d_transformed = W_d_transformed @ torch.linalg.pinv(T_d_to_b)
        
        # 确保维度完全匹配
        assert base_proj.weight.data.shape == W_d_transformed.shape, f"{proj_name} 维度不匹配"
        
        base_proj.weight.data = (1 - alpha) * base_proj.weight.data + alpha * W_d_transformed
        if base_proj.bias is not None and donor_proj.bias is not None:
            # bias 只受输出变换影响
            bias_d_transformed = T_d_to_b @ donor_proj.bias.data
            assert base_proj.bias.data.shape == bias_d_transformed.shape, f"{proj_name} bias 维度不匹配"
            base_proj.bias.data = (1 - alpha) * base_proj.bias.data + alpha * bias_d_transformed

    # 3. 变换输出投影层 o_proj
    base_o_proj = getattr(base_layer, 'o_proj')
    donor_o_proj = getattr(donor_layer, 'o_proj')
    
    # 同样应用双向变换
    W_d_transformed = T_d_to_b @ donor_o_proj.weight.data @ torch.linalg.pinv(T_d_to_b)
    
    assert base_o_proj.weight.data.shape == W_d_transformed.shape, "o_proj 维度不匹配"
    
    base_o_proj.weight.data = (1 - alpha) * base_o_proj.weight.data + alpha * W_d_transformed
    if base_o_proj.bias is not None and donor_o_proj.bias is not None:
        bias_d_transformed = T_d_to_b @ donor_o_proj.bias.data
        assert base_o_proj.bias.data.shape == bias_d_transformed.shape, "o_proj bias 维度不匹配"
        base_o_proj.bias.data = (1 - alpha) * base_o_proj.bias.data + alpha * bias_d_transformed

# --- 5. 主执行流程 ---

def main(alpha=0.5, alignment_type='LMA'):
    """执行模型对齐与合并的主函数"""
    tokenizer_donor, model_donor = load_complete_model(CKPT_PATH["llama2"], MODEL_DEVICE_A)
    tokenizer_base, model_base = load_complete_model(CKPT_PATH["qwen2"], MODEL_DEVICE_B)
    
    # 始终将层数多的作为deep，少的作为shallow
    if model_donor.config.num_hidden_layers >= model_base.config.num_hidden_layers:
        model_deep, model_shallow, tok_deep, tok_shallow = model_donor, model_base, tokenizer_donor, tokenizer_base
        deep_name, shallow_name, deep_device, shallow_device = "llama2", "qwen2", MODEL_DEVICE_A, MODEL_DEVICE_B
    else:
        model_deep, model_shallow, tok_deep, tok_shallow = model_base, model_donor, tokenizer_base, tokenizer_donor
        deep_name, shallow_name, deep_device, shallow_device = "qwen2", "llama2", MODEL_DEVICE_B, MODEL_DEVICE_A

    names_deep = [f"model.layers.{i}" for i in range(model_deep.config.num_hidden_layers)]
    names_shallow = [f"model.layers.{i}" for i in range(model_shallow.config.num_hidden_layers)]
    
    dataset = load_and_prepare_dataset(tok_deep, tok_shallow)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    print("\n--- 步骤 2: 为两个模型收集特征表示 ---")
    reps_deep, hooks_deep = register_hooks_for_reps(model_deep, names_deep)
    reps_shallow, hooks_shallow = register_hooks_for_reps(model_shallow, names_shallow)
    
    for batch in tqdm(dataloader, desc="特征提取"):
        with torch.no_grad():
            inputs_deep_data = batch[f"input_ids_{'a' if deep_name=='llama2' else 'b'}"].to(deep_device)
            inputs_shallow_data = batch[f"input_ids_{'a' if shallow_name=='llama2' else 'b'}"].to(shallow_device)
            model_deep(inputs_deep_data)
            model_shallow(inputs_shallow_data)
    
    for hook in hooks_deep + hooks_shallow: hook.remove()

    print(f"\n--- 步骤 3: 深度异构对齐 (使用 {alignment_type}) ---")
    cka_matrix = compute_cka_matrix(reps_deep, reps_shallow, names_deep, names_shallow)
    layer_alignment = align_layers_lma(cka_matrix)
    
    print("\n找到的层级映射关系 (Shallow Layer -> Deep Segment):")
    for i, segment in enumerate(layer_alignment):
        print(f"  {names_shallow[i]} -> {[names_deep[j] for j in segment]}")

    print(f"\n--- 步骤 4: 宽度异构合并 (alpha={alpha}) ---")
    merged_model = model_base

    for i, deep_segment_indices in enumerate(tqdm(layer_alignment, desc="合并所有层段")):
        shallow_layer_name = names_shallow[i]
        
        # 简化策略：使用段的最后一层激活作为整个段的代表
        last_deep_layer_name = names_deep[deep_segment_indices[-1]]
        reps_deep_segment = torch.cat(reps_deep[last_deep_layer_name], dim=0).flatten(0, 1)
        reps_shallow_layer = torch.cat(reps_shallow[shallow_layer_name], dim=0).flatten(0, 1)
        
        target_width = merged_model.config.hidden_size
        
        # 在CPU上执行ZipIt!
        Tm_s, Tm_d, Tu_s, Tu_d = match_tensors_zipit(reps_shallow_layer, reps_deep_segment, target_width)

        # 累积合并：将段内所有层的贡献融合到base模型的一层中
        base_target_layer = get_module_by_name(merged_model, shallow_layer_name)
        
        # 加权更新
        with torch.no_grad():
            for k, deep_layer_idx in enumerate(deep_segment_indices):
                donor_layer = get_module_by_name(model_deep, names_deep[deep_layer_idx])
                
                # 为了合并，我们将base_layer看作一个donor，它自己合并到自己身上
                apply_hetero_merge_to_layer_final(
                    base_target_layer.self_attn, 
                    donor_layer.self_attn, 
                    Tm_s, Tm_d, Tu_s, Tu_d, 
                    alpha / len(deep_segment_indices)
                )

    # --- 步骤 5: 保存并测试 ---
    output_dir = f"./merged_{shallow_name}_and_{deep_name}_alpha_{alpha}"
    print(f"\n--- 正在保存合并后的模型到 {output_dir} ---")
    merged_model.save_pretrained(output_dir)
    tok_shallow.save_pretrained(output_dir)
    print("模型保存完成。")

    del model_donor, model_base, reps_deep, reps_shallow, dataset, dataloader, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- 测试合并后的模型 ---")
    tokenizer_merged, model_merged = load_complete_model(output_dir, COMPUTE_DEVICE)
    prompt = "The capital of France is"
    inputs = tokenizer_merged(prompt, return_tensors="pt").to(COMPUTE_DEVICE)
    with torch.no_grad():
        outputs = model_merged.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer_merged.eos_token_id)
    print("输入:", prompt)
    print("合并后模型输出:", tokenizer_merged.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main(alpha=0.5)