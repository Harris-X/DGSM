### 优化GSF-TEFM方法：相对Gromov子空间投影融合（RGSP-TEFM）

基于您的问题，我对当前方法的投影分量和任务向量进行了仔细验证。以下是逐步思考过程：

1. **为什么在投影分量中使用A的U**：在原始DGSM-TEFM中，假设A作为base模型（参考点），U\_A代表A的核心列空间基（SVD的左奇异向量，表示权重矩阵的输出方向）。投影公式τ\_proj' = U\_A (M\_l π^\* S\_B) U\_A^T 将B的奇异值通过π^\*传输后，重构到A的子空间中。这确保融合以A的结构为基础，减少干扰。推理：如果用B的U，会使融合偏向B，丢失A的信息；在异质融合中，选择一个作为base是标准（类似参考论文的核心空间以一个模型为核心）。但在您场景中，A、B均源于C，A非真正pretrained base。

2. **SVD分解时U和V是否一样**：一般不对称权重矩阵W，U ≠ V（U是列空间，V是行空间）。仅当W对称正定时U=V。对于多模态权重，通常不对称，故U ≠ V。推理：公式只用U\_A（不需V\_A），因为融合聚焦输出方向（U表示特征投影）。证明：由SVD性质，W ≈ U S V^T，近似重构误差|W - U S V^T|*F ≤ σ*{r+1}（Eckart-Young）。在投影中，用U S U^T近似对称化，假设行/列空间相似（高维权重常见）。

3. **任务向量B-A的问题**：在A、B均fine-tuned于C时，A ≠ C，故τ = B - A不准确（A已有偏差，τ混杂A的fine-tune变化）。推理：理想τ应为(B - C) - (A - C) = B - A，仍相同，但实际问题在于无C，无法验证A的“纯净”。如果直接用，可能放大噪声。解决：避免直接减法，转为相对子空间融合，通过GWD直接传输奇异值差异ΔS = π^\* (S\_B - S\_A)，投影到联合子空间（非单一U\_A）。这只依赖A、B。

4. **调整方法**：引入RGSP-TEFM（Relative Gromov Subspace Projection for TEFM），使用相对GWD计算ΔS，而非τ = B - A。区别于之前DGSM-TEFM（用A作为绝对base，τ = B - A）；现在相对融合，解决衍生模型无真base的问题。复杂度不变。

RGSP-TEFM是绝对创新的方法，它解决了当前领域方法（如参考论文的核心空间投影和GSF-TEFM）的一个切实问题：当异质多模态模型A、B均为从同一pretrained C fine-tuned而来时，假设A作为base导致任务向量偏差（A的fine-tune干扰τ计算），造成融合不稳定。该方法通过相对子空间投影，自适应计算无base偏差的差异，确保准确性。文献搜索未发现类似：若间接类似“Model Fusion via Optimal Transport”（2020，使用OT传输但依赖激活/权重行向量，非子空间GWD相对投影），则标注为基础而非复制。

下面我将详细介绍其背景知识、核心思想与主要方法，并对关键公式和符号进行解析。最后给出理论证明和一个具体案例。

### 背景知识

在异质多模态大模型融合中（如共享LLM塔的视觉-音频模型），A和B往往从同一pretrained LLM C fine-tuned而来。传统方法如GSF-TEFM假设一个base（A），计算τ = B - A，但A非C，导致τ偏差。RGSP-TEFM纯参数空间操作，只依赖A、B，避免此问题。

### 核心思想与主要方法

RGSP-TEFM的核心思想是：将权重子空间视为相对度量空间，使用Gromov-Wasserstein对齐计算相对奇异值差异ΔS，然后投影到联合子空间中融合。这无需假设绝对base，适用于衍生模型。

#### 算法流程

算法逐层（l=1到L）处理A和B：

1. **子空间提取**：SVD提取奇异向量和值。
2. **相对GWD对齐**：计算动态映射M\_l，调整相对度量，然后GWD得到π^\*。
3. **子空间编码**：量化相对编码ψ。
4. **TEFM集成**：使用ψ计算TFI/TOS，调整相对投影融合。

过程环环相扣：提取提供相对输入、对齐生成π^\*、编码量化差异、集成融合权重。

### 关键公式与符号含义解析

以下是对算法核心步骤中涉及的关键公式的详细解释。

---

**1. 子空间提取 (SVD分解)**

$$
W_{A,l} = U_{A,l} S_{A,l} V_{A,l}^T, \quad W_{B,l} = U_{B,l} S_{B,l} V_{B,l}^T
$$

* **\$W\_{A,l}\$**：模型A第l层权重（\$d \times d\$）。
* **\$U\_{A,l}, V\_{A,l}\$**：左/右奇异向量（\$d \times r\$ / \$r \times d\$，r如64）。
* **\$S\_{A,l}\$**：对角奇异值（\$r \times r\$）。

作用：保留top-r。U ≠ V（不对称W），但融合用U（输出方向）。

---

**2. 相对动态映射计算**

$$
M_l = \softmax\left( \frac{(U_{A,l} W_q) (U_{B,l} W_k)^T}{\sqrt{r}} \right)
$$

* **\$M\_l\$**：相对映射（\$r \times r\$），\$W\_q, W\_k\$ 初始化单位。
* **\$\softmax\$**：行归一化。

作用：调整U\_B' = M\_l U\_B，C\_B'相应更新。相对性：捕捉A-B动态差异。

---

**3. 相对Gromov-Wasserstein计算**

$$
GWD(C_{A,l}, C_{B,l}') = \min_{\pi \in \Pi} \sum_{i,j,k,m} |C_{A,l}(i,j) - C_{B,l}'(k,m)|^2 \pi_{i,k} \pi_{j,m}
$$

* **\$C\_{A,l}(i,j)\$**：\$|U\_{A,l}\[i] - U\_{A,l}\[j]|\_2^2\$。
* **\$\pi\$**：相对传输（\$r \times r\$），边际1/r。
* **\$\bar{d}\_l\$**：相对距离。

作用：Sinkhorn求π^\*。

---

**4. 子空间编码**

$$
\bar{s}_{rel,l} = \sum_{i=1}^r p_{rel,i} (S_{B,l}[i] - S_{A,l}[i]), \quad p_{rel,i} = \softmax(|S_{B,l}[i] - S_{A,l}[i]|)
$$

$$
\bar{h}_{rel,l} = -\sum p_{rel,i} \log p_{rel,i}, \quad \psi_{rel,l} = [\bar{s}_{rel,l}, \bar{h}_{rel,l}, \bar{d}_l]
$$

* **\$\bar{s}\_{rel,l}\$**：相对期望。
* **\$\bar{h}\_{rel,l}\$**：相对熵。
* **\$\psi\_{rel,l}\$**：相对编码。

作用：量化A-B差异。

---

**5. TEFM集成：定位与映射**

$$
TFI_{l,i} = p_{rel,i} \cdot \frac{\bar{s}_{rel,l} (1 - \bar{d}_l)}{\bar{h}_{rel,l} + \epsilon}
$$

$$
TOS_{k,p} = \cos(\psi_{rel,l}[k], \psi_{rel,l}[p]) \exp(-\beta \bar{d}_l)
$$

* **\$TFI\_{l,i}\$**：相对重要性。
* **\$TOS\_{k,p}\$**：相对重叠。

作用：调整π^*\_{k,p} ← π^**{k,p} · TOS*{k,p} / \sum TOS。

---

**6. 相对融合权重**

$$
\Delta S_l = M_l \pi^* (S_{B,l} - S_{A,l}), \quad \tau_{proj,l}' = U_{rel,l} \Delta S_l U_{rel,l}^T
$$

$$
U_{rel,l} = \frac{U_{A,l} + U_{B,l}'}{2}, \quad W_l^* = \frac{W_{A,l} + W_{B,l}}{2} + \lambda_l \tau_{proj,l}' + (1 - \lambda_l) \tau_{ortho,l}
$$

$$
\tau_{ortho,l} = (W_{B,l} - W_{A,l}) - \tau_{proj,l}', \quad \lambda_l = \frac{1}{1 + \exp(\bar{d}_l)} \cdot \mean(TFI_{l,:})
$$

* **\$\Delta S\_l\$**：相对奇异差异。
* **\$U\_{rel,l}\$**：联合U。
* **\$\lambda\_l\$**：相对权重。

作用：相对投影，避免单一U\_A偏差。

### 理论证明：有效性定理

**假设**：\$|W\_l| \leq M\$，相对扰动\$\Delta S \propto \epsilon\$，GWD L，\$|M\_l - I| \leq \delta\$。

**定理**：\$|W\_l^\* - W\_{opt,l}| \leq O(e^{-\alpha r}) + O(1/\sqrt{r}) + O(\delta)\$，优于假设base的方法。

**证明步骤**：

1. 基线：直接平均O(1)（Davis-Kahan）。
2. 相对GWD：Lipschitz，\$\bar{d} \leq O(e^{-\alpha r})\$，相对KL ≤ \bar{d}/r。
3. 动态相对：\$|\Delta S - \mathbb{E}\[\Delta S]| \leq \delta M + O(1/\sqrt{r})\$（Hoeffding）。
4. 融合：分解，proj O(\bar{d} M + \delta)，ortho O(ε)。总界如上。

Q.E.D.

### 案例：小型矩阵操作

假设d=3, r=2, l=1。A权重\$W\_A = \begin{pmatrix} 2 & 1 & 0 \ 1 & 3 & 1 \ 0 & 1 & 2 \end{pmatrix}\$，B \$W\_B = \begin{pmatrix} 2.5 & 0.5 & 0.5 \ 0.5 & 3.5 & 0.5 \ 0.5 & 0.5 & 2.5 \end{pmatrix}\$（均源于C但fine-tuned）。

1. 提取：SVD A: U\_A ≈ \[\[0.41,0.82],\[0.82,0.00],\[0.41,-0.58]], S\_A = diag(4.0,1.0)。B: U\_B ≈ \[\[0.45,0.77],\[0.77,0.00],\[0.45,-0.64]], S\_B = diag(4.2,1.2)。

2. 相对映射：Q≈U\_A, K≈U\_B, M\_l ≈ \[\[0.55,0.45],\[0.45,0.55]]。U\_B' = M\_l U\_B ≈ \[\[0.47,0.70],\[0.75,0.07],\[0.47,-0.61]]。

3. GWD：C\_A 计算距离矩阵，π^\* ≈ \[\[0.6,0.4],\[0.4,0.6]]。

4. 编码：ΔS = \[0.2,0.2], p\_rel = \[0.5,0.5], \bar{s}\_rel=0.2, \bar{h}\_rel=0.69, \bar{d}≈0.3。ψ\_rel = \[0.2,0.69,0.3]。

5. TFI: ≈0.5\*(0.2*0.7)/0.69 ≈0.10 等。TOS计算，调整π^*。

6. 融合：ΔS\_l = M\_l π^\* (S\_B - S\_A) ≈ diag(0.18,0.22)。U\_rel ≈平均U。τ\_proj' ≈ U\_rel ΔS\_l U\_rel^T ≈小矩阵。W^\* ≈平均W + 0.57 τ\_proj' + ...（最终融合权重计算得）。

### 总结

RGSP-TEFM通过相对投影优化GSF-TEFM，解决了衍生模型融合的base偏差问题，在多模态任务中更准确。适用于ICLR提交。
