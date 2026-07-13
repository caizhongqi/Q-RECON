# Q-RECON：研究创新与 CCF-A 论文路线

## 1. 问题定义

设训练算法为

\[
\theta=A(D),
\]

其中 \(D\) 是私有训练集，\(\theta\) 是训练后的经典模型参数。攻击者通过观测接口 \(\mathcal O\) 得到

\[
o=\mathcal O(\theta,D),
\]

并尝试恢复

\[
\hat D=R(o).
\]

精确恢复的必要条件是：在给定候选空间和攻击者先验下，映射 \(D\mapsto o\) 至少在目标等价类上具有可识别性。若存在

\[
D_1\ne D_2,\qquad \mathcal O(A(D_1),D_1)=\mathcal O(A(D_2),D_2),
\]

则任何经典或量子攻击者都无法仅凭该观测判断真实训练集。

## 2. 量子力学与 QNN 基础

主流量子神经网络采用参数化量子电路：

\[
\rho(x)=U_{\mathrm{enc}}(x)|0\rangle\langle0|U_{\mathrm{enc}}^\dagger(x),
\]

\[
f_\phi(x)=\operatorname{Tr}
\left[O U_\phi\rho(x)U_\phi^\dagger\right].
\]

量子演化本身保持线性，QNN 的有效非线性主要来自重复数据编码、测量和经典反馈。量子搜索或量子梯度优势依赖相干查询，而普通模型 API 只提供经典输入与经典输出。

普通接口为

\[
x\mapsto f_\theta(x),
\]

量子查询算法需要的接口为

\[
U_f|x\rangle|y\rangle
=|x\rangle|y\oplus Q_b(f_\theta(x))\rangle,
\]

其中 \(Q_b\) 表示 \(b\) 位定点量化。

## 3. 三类访问模型

### 3.1 C-Access：经典黑盒访问

攻击者提交经典输入并获得硬标签或 logits。每次查询都会得到经典结果，不保留候选输入之间的量子相干性。

### 3.2 W-Access：白盒或训练过程访问

攻击者获得以下一种或多种信息：

- 模型参数；
- 单样本或小批量梯度；
- 参数更新；
- 中间层激活；
- 优化器状态。

该场景适用于联邦学习、分割学习和模型发布场景。

### 3.3 Q-Access：相干量子访问

目标经典模型被转换为可逆量子电路，允许对输入叠加态进行相干查询。只有在这一访问模型下，才能严格讨论 Grover 搜索、幅度估计和量子梯度的查询复杂度优势。

## 4. 核心创新点

### 创新一：访问模型能力分离与不可恢复定理

建立 C-Access、W-Access 和 Q-Access 的形式化定义，并研究：

1. 普通经典 API 为什么不能直接提供依赖相干查询的量子查询优势；
2. 在 Q-Access 下，有限候选空间是否存在平方级查询加速；
3. 当多个候选训练集产生相同观测时，最优恢复成功率的上界；
4. 观测噪声、量化精度和差分隐私对可恢复性的影响。

对于两个候选训练集对应的量子态 \(\rho_{D_0}\) 和 \(\rho_{D_1}\)，可使用 Helstrom 界：

\[
P_e^*=\frac12\left(1-\frac12
\|\rho_{D_0}-\rho_{D_1}\|_1\right).
\]

当两个状态完全相同时，量子攻击者也只能随机猜测。

### 创新二：经典神经网络的相干预言机编译

设计从量化模型到可逆量子电路的自动编译流程：

- 定点乘加；
- ReLU 和比较器；
- 卷积权重复用；
- logit 量化；
- 辅助量子位清理；
- 中间计算的 uncomputation；
- T-count、逻辑量子位与电路深度估算。

目标不是仅证明“可以编译”，而是推导预言机复杂度，并判断其是否抵消量子搜索收益。

### 创新三：可识别性感知的量子训练数据重构

定义统一重构能量：

\[
E(D')=
\|\mathcal O(A(D'))-o\|^2+\lambda R(D').
\]

将 \(E\) 编译为 phase oracle、候选验证器或 Ising Hamiltonian。为避免直接搜索高维像素空间，使用结构化生成先验：

\[
D'=G(z),\qquad z\in\{0,1\}^m.
\]

若候选空间大小为 \(N\)，满足约束的候选数为 \(K\)，研究是否能够实现

\[
Q_{\mathrm{quantum}}=\Theta(\sqrt{N/K})
\]

相对于经典无结构搜索的查询优势。若 \(K>1\)，算法必须输出等价解集合或后验分布，不能把任意可行解宣称为原始训练样本。

### 创新四：表达能力—可训练性—可恢复性三角关系

利用动力学李代数、梯度方差和量子 Fisher 信息研究攻击 QNN：

\[
\text{Expressivity}
\leftrightarrow
\text{Trainability}
\leftrightarrow
\text{Recoverability}.
\]

目标是构造泄漏感知的最小充分李代数：

\[
\mathfrak g_{\mathrm{attack}}
=\operatorname{Lie}
\{H_{\mathrm{enc}},H_{\mathrm{leak}},H_{\mathrm{prior}}\}.
\]

该结构应同时满足：

- 李代数维度为多项式规模；
- 梯度方差不指数消失；
- 对目标候选空间保持可分性；
- 测量与参数数量低于通用 hardware-efficient ansatz。

### 创新五：局部—全局多尺度量子恢复损失

针对量子生成模型的可训练性与高阶可区分性冲突，采用

\[
L_t=\alpha_tL_{\mathrm{local}}
+\beta_tL_{\mathrm{global}}
+\gamma_tL_{\mathrm{leak}}.
\]

- 局部低体可观测量保证早期梯度；
- 全局 witness 排除只匹配局部统计的假样本；
- 泄漏一致性项匹配梯度、参数、激活或 logits；
- 训练过程中逐步提高全局损失权重。

## 5. 建议的一篇主论文

主论文收敛到三个主要贡献：

1. 量子访问模型与不可恢复边界；
2. 经典神经网络到相干预言机的编译；
3. 可识别条件下具有查询优势的量子重构算法。

建议英文题目：

> Q-RECON: Quantum Access Models, Coherent Oracles, and Identifiability Limits for Training-Data Reconstruction

## 6. 实验设计

### 6.1 目标模型

- Logistic Regression；
- MLP；
- CNN；
- Tiny Transformer；
- GNN 或 Random Forest。

### 6.2 数据模态

- 表格数据，用于验证精确恢复；
- 图像数据，在生成模型潜空间进行搜索；
- 离散 token 或图结构数据。

### 6.3 访问条件

- 硬标签；
- logits；
- 完整参数；
- 单样本和小批量梯度；
- 编译后的相干量子预言机。

### 6.4 基线

- DLG 与梯度反演改进方法；
- 可证明的张量分解重构；
- GMI、KEDMI、PLG-MI、BREP-MI；
- 经典遗传算法、模拟退火和贝叶斯优化；
- quantum-inspired 方法；
- 相同参数量的经典生成器。

### 6.5 评价指标

- Exact Match；
- 最近真实训练样本距离；
- 假阳性率；
- SSIM、LPIPS 或 token edit distance；
- 经典与量子查询次数；
- 逻辑量子位；
- T-count；
- 电路深度；
- shots；
- 包含编码和读出的端到端成本。

## 7. 投稿定位

- ICML/NeurIPS：可识别性、查询复杂度和量子学习理论；
- IEEE S&P/ACM CCS/USENIX Security/NDSS：隐私泄漏、真实威胁模型和端到端攻击；
- ASPLOS/ISCA/MICRO：预言机编译和量子资源优化；
- CVPR/ICCV：仅当工作核心转向高保真视觉训练样本恢复。

仅展示 QNN 在小型数据集上比经典模型提高少量重构指标，不足以形成顶会级贡献。论文应同时包含能力边界、正向算法结果、资源核算和严格的原始样本真实性评价。

## 8. 关键参考文献

1. [Barren plateaus in variational quantum computing](https://www.nature.com/articles/s42254-025-00813-9)
2. [Theoretical guarantees for permutation-equivariant quantum neural networks](https://www.nature.com/articles/s41534-024-00804-1)
3. [Trainability barriers and opportunities in quantum generative modeling](https://www.nature.com/articles/s41534-024-00902-0)
4. [Characterizing privacy in quantum machine learning](https://www.nature.com/articles/s41534-025-01022-z)
5. [Optimizing quantum optimization algorithms via faster quantum gradient computation](https://arxiv.org/abs/1711.00465)
6. [Circuit complexity of quantum access models for encoding classical data](https://arxiv.org/abs/2311.11365)
7. [Xor-And-Inverter Graphs for Quantum Compilation](https://www.nature.com/articles/s41534-021-00514-y)
8. [Data Reconstruction Attacks and Defenses: A Systematic Evaluation](https://proceedings.mlr.press/v258/liu25b.html)
9. [Data Reconstruction: Identifiability and Optimization with Sample Splitting](https://arxiv.org/abs/2602.08723)

