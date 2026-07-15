# Q-RECON

面向经典机器学习模型训练数据重构的量子访问理论、相干预言机与可识别性研究。

## 研究目标

本项目不把量子神经网络简单视为经典优化器的替代品，而是从量子信息与量子查询模型出发，研究三个基础问题：

1. 经典模型中的训练数据在什么条件下仍然可被唯一识别？
2. 普通经典 API 与相干量子预言机之间是否存在严格的攻击能力分离？
3. 在预言机实现成本计入总成本后，训练数据重构能否获得可证明的量子查询优势？

## 核心贡献设想

- 建立经典访问、白盒访问与相干量子访问三类威胁模型。
- 给出训练数据不可识别时的经典/量子信息论上界。
- 将量化经典神经网络和训练泄漏目标编译为干净的可逆量子预言机。
- 在唯一可识别的结构化候选空间中设计量子重构算法。
- 证明并测量预言机构造、近似误差和端到端成本下的优势或无优势边界。
- 建立覆盖多种模型、数据模态和访问方式的统一基准。

## 文档

- [形式化理论基础、定理与证明](docs/THEORY_FOUNDATIONS.md)
- [理论主张矩阵与端到端优势准入条件](docs/THEORY_CLAIM_MATRIX.md)
- [CCF-A 投稿准备度与内部拒绝规则](docs/CCF_A_READINESS.md)
- [统计基准、置信区间与证据质量协议](docs/STATISTICAL_BENCHMARK_PROTOCOL.md)
- [可哈希清单、重复计时与层级统计](docs/MANIFEST_DRIVEN_BENCHMARKS.md)
- [真实数据双记录梯度 fibre、two-sum 与优势/无优势相图](docs/REAL_DATA_BATCH_GRADIENT_PHASE_DIAGRAM.md)
- [量化候选的碰撞、失真与加载审计](docs/QUANTIZED_CANDIDATE_AUDIT.md)
- [聚合梯度的批次不可识别性定理](docs/BATCH_GRADIENT_NONIDENTIFIABILITY.md)
- [单样本完整训练梯度的可识别性与相干重构预言机](docs/GRADIENT_RECONSTRUCTION_ORACLE.md)
- [干净聚合梯度重构预言机与可识别性区间](docs/BATCH_GRADIENT_ORACLE.md)
- [相干预言机编译器规范](docs/COHERENT_ORACLE_SPEC.md)
- [精确 truth-table 相干预言机基线](docs/TRUTH_TABLE_ORACLE_BASELINE.md)
- [ANF 精确预言机综合优化](docs/ANF_ORACLE_OPTIMIZATION.md)
- [保持结构的可逆 Affine 编译器](docs/STRUCTURE_PRESERVING_AFFINE_ORACLE.md)
- [保持结构的可逆两层 MLP/ReLU 编译器](docs/REVERSIBLE_MLP_ORACLE.md)
- [任意深度可逆整数 ReLU MLP 与共享工作区定理](docs/DEEP_REVERSIBLE_MLP_ORACLE.md)
- [任意深度定点 Affine/ReLU value、threshold 与 equality 预言机](docs/FIXED_POINT_DEEP_MLP_ORACLE.md)
- [固定点两层 MLP 精确观测相等预言机](docs/FIXED_POINT_MLP_EXACT_OBSERVATION.md)
- [结构化有限乘积域的干净 membership 与组合预言机](docs/STRUCTURED_DOMAIN_ORACLE.md)
- [Z3 固定点 MLP 精确反演基线](docs/Z3_FIXED_POINT_INVERSION.md)
- [未知标记数下的 BBHT 搜索与有限成功率证书](docs/UNKNOWN_K_QUANTUM_SEARCH.md)
- [BBHT 零解的一侧错误判定与查询成本](docs/UNKNOWN_K_ZERO_SOLUTION.md)
- [未知标记数下的稳健端到端成本比较](docs/UNKNOWN_K_END_TO_END_COSTING.md)
- [显式经验候选表的数据加载无优势证书](docs/EXPLICIT_DATA_LOADING_NO_ADVANTAGE.md)
- [端到端成本比较协议](docs/END_TO_END_COST_PROTOCOL.md)
- [理论到实验的评估协议](docs/THEORY_EVALUATION_PROTOCOL.md)
- [研究创新与论文路线](docs/RESEARCH_INNOVATION.md)
- [数据集与下载方式](docs/DATASETS.md)
- [本地验证记录](docs/VERIFICATION.md)
- [解析恢复的适用条件](docs/RECOVERY_ASSUMPTIONS.md)

## 已实现功能

- GIFT-Eval 时间序列流式加载器；
- TIME 2026 本地数据适配器；
- Community Forensics Small 图像流式加载器；
- GIFT-Eval 与 Community Forensics 的 revision-pinned 候选清单和选中特征/目标 SHA256 锁定；
- CLOFAI/通用 ImageFolder 适配器；
- 时间序列 MLP 与图像 CNN 受害模型；
- 单样本梯度泄漏与梯度反演攻击；
- 首层 Linear 单样本梯度的解析精确恢复；
- iDLG 风格的分类标签自动推断；
- 直接、经典生成和变分量子三类重构先验；
- 基于完整梯度 Jacobian 秩的局部可识别性分析；
- 确定性 fibre 与噪声观测通道的 Bayes 最优恢复上界；
- 面向声明等价关系的 Bayes 最优恢复上界；
- 数据处理不等式、条件 min-entropy 与二元 Helstrom 界的可执行实现；
- batch size ≥ 2 的带偏置线性回归聚合梯度连续碰撞构造；
- 单样本带偏置线性回归完整梯度的“非零残差可解析恢复／零残差不可识别”二分定理；
- 公开标签小候选域上的聚合梯度全局可识别证书，以及私有标签下超出批次置换的非平凡碰撞 fibre；
- 双记录加性梯度的精确二分边界：`K>1` 时原始批次信息论不可识别，`K=1` 时向量 two-sum 与 Grover 均为候选数的线性指数；
- 真实/合成候选的量化碰撞、全局 pair fibre、条件 Bayes 上限、完整 two-sum 解集与失败精度保留；
- 经典无放回搜索、已知 `K` 标准 Grover 与未知 `K` BBHT 随机迭代成功率/查询模型；
- 对所有允许正标记数逐一核验的 BBHT 有限空间统一成功率证书，并单独报告 phase 与测量后验证查询；
- 精确 verifier 下的 BBHT 零解一侧错误判定：`K=0` 时无假阳性，正标记数继承统一检测率；
- 未知 `K` 搜索的最坏期望量子成本包络，以及与任意声明的最强专用经典求解器的同单位比较；
- 端到端成本 break-even、量子搜索计划优化与近似预言机误差界；
- 显式经验候选表的 `Nw` 编译 bit-probe 下界、典型电路下界、minterm 上界、摊销门槛与可执行无优势 workload 区间；
- 二进制定点、双补码、确定性舍入、显式溢出语义与逐层区间证明；
- 量化 Logistic Regression/MLP 的 bit-exact 参考求值器；
- 有限候选空间上的干净 truth-table value oracle、verifier 与 phase oracle；
- GF(2) 代数标准形（ANF）精确 oracle 综合与资源感知后端选择；
- 多项式规模的可逆整数 Affine value/threshold oracle：常数 shift-add、ripple-carry、copy 与严格反计算；
- 可逆两层 `Affine → ReLU → Affine/Threshold` MLP 相位预言机；
- 任意隐藏层深度的整数 ReLU MLP 相位预言机；
- 任意深度定点 Affine/ReLU 网络的 clean value、threshold 与多输出 exact-equality phase oracle；
- 任意深度定点编译的逐层可达性证书、隐藏寄存器 Bennett 清理、共享算术工作区与精确 `(2,…,2,1)` 层调用恒等式；
- 非连续有符号固定点乘积域的 clean feature-membership、全特征 AND 与 MLP exact-output 组合预言机；
- 与 fixed-point MLP exact-equality oracle 完全同语义的完备 branch-and-bound 经典反演器；
- 与参考求值器相同舍入、ReLU、饱和和溢出语义的 Z3 SMT 精确反演器及 complete/incomplete 终止证明；
- branch-and-bound、SMT 与 domain-restricted coherent oracle 的三方解集一致性审计；
- 多配置/多种子 benchmark 的确定性 percentile-bootstrap、Wilson 区间、log-log 规模拟合、重复种子拒绝与可机读质量门；
- 规范化实验清单 SHA256、预热和重复计时、异常类型/消息哈希、重复语义一致性校验、按种子中位数折叠与跨种子层级统计；
- 跨层共享算术工作区，峰值 ancilla 由最大单层工作量而非各层之和决定；
- ReLU 的双补码符号控制 Toffoli 实现及隐藏激活/预激活的 Bennett 清理；
- 单样本训练梯度的保持结构算术预言机：可逆残差、可逆有符号变量乘法、全梯度输出和精确相等 verifier；
- 有序批次聚合梯度的保持结构算术预言机：逐记录共享工作区、模加聚合、全梯度输出、精确相等 verifier 与完全反计算；
- truth-table、ANF、Affine、MLP、fixed-point MLP、domain membership、单样本梯度与批次梯度多后端的逐输入等价、自逆、置换和 ancilla 清理验证；
- 有限空间全局 fibre、碰撞规模和 Bayes 恢复上限分析；
- 由真实编译 predicate gate netlist 驱动的 Grover 状态向量验证；
- 可机读的 logical-qubit、ancilla、controlled-X、Toffoli、T-count、T-depth、查询和摊销成本报告；
- parity、all-zero equality、majority、Affine、MLP、single-gradient 与 batch-gradient predicate 的综合扩展分析；
- 重构质量与量子逻辑资源统计；
- Python 3.10/3.12 理论与编译器 CI、独立 Z3 solver CI、统计/清单报告归档，以及 PennyLane 前向反向 smoke test；
- YAML/JSON 实验配置和单元测试。

## 安装

建议使用 Python 3.10 以上版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[quantum,solver,test]'
```

## 快速验证

```bash
qrecon --config configs/smoke.yaml
pytest
python examples/theory_bounds.py
python examples/coherent_oracle_demo.py
python examples/oracle_scaling.py
python examples/affine_oracle_cost_report.py
python examples/mlp_oracle_demo.py
python examples/gradient_reconstruction_demo.py
python examples/batch_gradient_demo.py
python examples/real_batch_gradient_phase_diagram.py
python examples/fixed_point_mlp_exact_observation.py
python examples/unknown_k_search.py
python examples/unknown_k_cost_envelope.py
python examples/explicit_data_loading_boundary.py
python examples/z3_fixed_point_inversion.py
python examples/fixed_point_benchmark_matrix.py
python examples/fixed_point_statistical_report.py
python examples/fixed_point_manifest_report.py
```

运行真实数据实验：

```bash
qrecon --config configs/time_gifteval_quantum.yaml
qrecon --config configs/image_community_forensics.yaml
qrecon --config configs/time_gifteval_analytic.yaml
qrecon --config configs/image_community_forensics_analytic.yaml
qrecon --config configs/image_community_forensics_lenet_lbfgs.yaml
python examples/real_batch_gradient_phase_diagram.py \
  --manifest configs/real_candidates/gifteval_batch2.json
python examples/real_batch_gradient_phase_diagram.py \
  --manifest configs/real_candidates/community_forensics_batch2.json
```

实验结果写入 `outputs/`，包括 `report.json` 和重构张量
`reconstruction.pt`；相图示例把完整 JSON 报告写到标准输出，建议由调用方重定向并归档。

## 当前状态

项目已经建立信息论恢复界、目标等价类恢复、局部与有限空间全局可识别性、聚合梯度显式碰撞族、理想与未知标记数查询复杂度、零解一侧错误判定、近似预言机误差、显式候选表加载下界以及端到端摊销成本的形式化基础。

相干编译部分具备多类可交叉验证的路径：mixed-polarity minterm、GF(2) ANF、保持结构的整数 Affine、任意深度整数 ReLU MLP、任意深度定点 Affine/ReLU value/threshold/equality、结构化 product-domain membership、单样本完整训练梯度，以及有序批次聚合梯度。枚举型后端提供有限空间独立精确综合基线；保持结构的后端执行真实 X/CNOT/Toffoli gate netlist，并通过 compute-copy-uncompute、反向层清理、逐记录反计算和共享工作区复用将全部算术、预激活、激活、残差、乘法、比较、域 membership 和聚合寄存器恢复为零。小规模配置对所有候选、两个初始目标位、逆电路、phase sign、Grover 曲线、碰撞 fibre、经典/SMT 解集一致性和资源恒等式进行穷举验证。

真实候选双记录梯度路径现在形成“revision-pinned 数据清单 → 选中特征/目标哈希 → 多精度量化与碰撞审计 → 全局 pair fibre → 完整向量 two-sum → 枚举型相干语义参考 → 未知 `K` 证书 → 显式加载摊销边界”的闭环。其核心结论是一个严格二分：目标 fibre 非唯一时原始索引批次不可识别；目标 fibre 唯一时，经典 two-sum 与理想 Grover 已具有相同的线性查询/时间指数，因此不能把对所有候选对的二次扫描作为经典基线来宣称新的平方加速。

项目仍不宣称已经获得实际端到端量子优势。下一门槛不再是补齐双记录加性任务，而是选择一个没有廉价解析或 two-sum 分解、仍可全局识别的真实泄漏任务，完成非枚举相干访问、最强 SAT/SMT/MIP/代数/连续经典求解器、固定硬件重复实验与故障容错成本相图；另一条同样有效的顶会路线，是把当前二分推广为覆盖更广模型与访问条件的严格无优势边界。当前 VQC 仍只是潜空间重构先验，不是上述相干训练泄漏预言机。

在 batch size 1、完整梯度可见且首层带偏置 Linear 直接接收原始输入时，解析攻击已在真实 GIFT-Eval 与 Community Forensics 样本上实现 `within 1e-6 = 100%`；该结论不适用于任意 CNN、聚合梯度或受防御保护的训练过程。
