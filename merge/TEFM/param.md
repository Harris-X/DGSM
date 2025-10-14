参数说明（按出现顺序）
模型 / 数据集选择：

--model: 使用 VLMEvalKit 中 supported_VLM 名称（本地权重、多模态模型）。
--hf-llm-id: 使用 transformers 纯文本 CausalLM（HF 名称或本地路径）。与 --model 互斥，二选一。
--data: 走 VLMEvalKit 的内置数据集加载（如 MME、MMBench 等），当未指定 --hf-dataset 时有效。
--hf-dataset: 特殊 HF 组合数据集开关，目前支持 'meta'（混合多数据集），指定后覆盖 --data。
Meta 组合数据集采样控制：

--n-mmbench: MMBench (en/test) 采样条数。
--n-vcr: VCR (validation Q->A) 采样条数。
--n-docvqa: DocVQA (validation) 采样条数。
--n-vqa: VQAv2 (validation) 采样条数。
--n-scienceqa: ScienceQA (validation, 有图) 采样条数。
--n-stvqa: ST-VQA task1 (test) 采样条数。
--max-samples: 对最终合并后的样本总数做上限截断（减少耗时）。
HF 下载 / 缓存 / 离线控制：

--hf-no-streaming: 禁用 streaming 模式（全部下载到本地后访问）。
--hf-endpoint: 指定或切换 HF Hub 域名（镜像），传 disable 取消覆盖。
--hf-offline: 强制离线（只用本地缓存）；会自动关闭 streaming。
--hf-cache-dir: 重定向 HF 缓存目录（所有相关环境变量一起指向，便于磁盘管理）。
--hf-disable-streaming-fallback: 遇到磁盘不足时不自动 fallback 到 streaming。
Hook 过滤：

--req-act: 记录哪类激活（input / output）。可两者都选。
--module-regex: 正则匹配模块全名（named_modules() 路径）——主筛选器。
--include-types: 限定模块类型类名（白名单），为空则不过滤类型。
--exclude-regex: 排除匹配的模块（黑名单）。
杂项：

--work-dir: 用于保存临时图片（HF meta 图像转文件），默认当前目录。
--save: 输出激活结果（平均值）的主 .pt 文件路径；若不指定，则放入 activations 自动命名。
--verbose: 打印匹配模块等详细信息。
--use-vllm: 对部分特定模型（如 Llama-4 / Qwen2-VL 系列）启用 vLLM 推理路径（兼容性取决于 wrapper）。
GPU / 设备：

--gpus: 逗号分隔 GPU id，内部设置 CUDA_VISIBLE_DEVICES。
--llm-device: 纯文本 LLM 不使用 device_map 时的放置设备（默认 cuda）。
--llm-dtype: LLM 权重加载精度（auto | float16 | bfloat16 | float32）。
--trust-remote-code: 允许执行自定义模型代码（HF trust_remote_code=True）。
--probe-batch-size: LLM 文本批量前向批大小（LAPE 关闭时批处理；LAPE 开启时逐条多路径）。
--llm-max-length: Tokenizer 截断最大长度。
--llm-device-map: transformers 的 device_map （多卡分片：'auto' / 'balanced' / ... / None）。
--llm-forward-mode: LLM 前向模式：generate（触发生成）或 forward（直接前向，适合只需激活）。
--llm-new-tokens: 在 generate 模式下每条样本生成的新 token 数（非 LAPE 情况下使用）。
VLM 分片（针对 VLMEvalKit 包装内部模型）：

--vlm-device-map: 若为 'auto' 则用 accelerate 推断并分片；否则默认 wrapper 自己的单卡行为。
--vlm-dtype: 推断分片时用于平衡内存的 dtype（float16/bfloat16/float32）。
--vlm-no-split-classes: accelerate 自动拆分时不要拆的类名列表（避免层被断开）。
FAI（Flatness-Activation Importance）相关：

--fai-compute: 启用 FAI 计算（需要保留批次激活）。
--fai-max-samples-per-module: 每个模块用于统计的最大行数（子采样，防爆内存）。
--fai-mi-mode: 估计依赖性的方式（mi sklearn mutual_info_regression；pearson；spearman）。
--fai-eps: FAI 分母稳定项 epsilon。
LAPE（Length-Agnostic Probabilistic Encoding）Stage-1：

--lape-enable: 启用 LAPE（多路径采样 + φ 统计）。
--lape-samples: 每个输入样本采样的生成路径数 N_s。
--lape-gamma: 路径长度折扣因子 γ（减弱长路径偏差）。
--lape-top-p: nucleus sampling 的 top-p。
--lape-temperature: 采样温度（统一控制不同模型）。
--lape-min-new: 每条路径最少生成 token。
--lape-max-new: 每条路径最多生成 token（随机在区间内抽）。
--lape-yref: 梯度代理参考：zero 或 running_avg（后者逐路径更新平滑）。
LAPE per-sample MI 支持（用于 Stage-2 真正 MI）：

--lape-mi-store: 保存每个样本的 φ 分量（phi_a/phi_g/phi_L）样本序列，用于后续互信息估计。
--lape-mi-max-samples: 每个模块最多保留多少输入样本的 φ（截断策略；-1 不限；0 禁用）。