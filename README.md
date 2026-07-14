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
- 将量化经典神经网络自动编译为干净的可逆量子预言机。
- 在唯一可识别的结构化候选空间中设计量子重构算法。
- 证明并测量预言机构造、近似误差和端到端成本下的优势边界。
- 建立覆盖多种模型、数据模态和访问方式的统一基准。

## 文档

- [形式化理论基础、定理与证明](docs/THEORY_FOUNDATIONS.md)
- [理论主张矩阵与端到端优势准入条件](docs/THEORY_CLAIM_MATRIX.md)
- [相干预言机编译器规范](docs/COHERENT_ORACLE_SPEC.md)
- [精确 truth-table 相干预言机基线](docs/TRUTH_TABLE_ORACLE_BASELINE.md)
- [理论到实验的评估协议](docs/THEORY_EVALUATION_PROTOCOL.md)
- [研究创新与论文路线](docs/RESEARCH_INNOVATION.md)
- [数据集与下载方式](docs/DATASETS.md)
- [本地验证记录](docs/VERIFICATION.md)
- [解析恢复的适用条件](docs/RECOVERY_ASSUMPTIONS.md)

## 已实现功能

- GIFT-Eval 时间序列流式加载器；
- TIME 2026 本地数据适配器；
- Community Forensics Small 图像流式加载器；
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
- 经典无放回搜索与标准 Grover 成功率/查询数模型；
- 端到端成本 break-even 与近似预言机误差界；
- 二进制定点、双补码、确定性舍入、显式溢出语义与逐层区间证明；
- 量化 Logistic Regression/MLP 的 bit-exact 参考求值器；
- 有限候选空间上的干净 truth-table value oracle、verifier 与 phase oracle；
- oracle 自逆/置换/ancilla 清理的穷举验证；
- 有限空间全局 fibre、碰撞规模和 Bayes 恢复上限分析；
- 由编译 predicate 驱动的 Grover 状态向量验证；
- 可机读的 logical-qubit、ancilla、Toffoli、T-count、T-depth 与查询资源报告；
- 重构质量与量子逻辑资源统计；
- YAML 实验配置和单元测试。

## 安装

建议使用 Python 3.10 以上版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[quantum,test]'
```

## 快速验证

```bash
qrecon --config configs/smoke.yaml
pytest
python examples/theory_bounds.py
python examples/coherent_oracle_demo.py
```

运行真实数据实验：

```bash
qrecon --config configs/time_gifteval_quantum.yaml
qrecon --config configs/image_community_forensics.yaml
qrecon --config configs/time_gifteval_analytic.yaml
qrecon --config configs/image_community_forensics_analytic.yaml
qrecon --config configs/image_community_forensics_lenet_lbfgs.yaml
```

实验结果写入 `outputs/`，包括 `report.json` 和重构张量
`reconstruction.pt`。

## 当前状态

项目已经建立信息论恢复界、目标等价类恢复、局部与有限空间全局可识别性、理想查询复杂度、近似预言机误差以及端到端成本的形式化基础。仓库现包含第一个可执行相干预言机里程碑：量化 Logistic Regression/MLP 的 bit-exact 参考语义能够被枚举为干净 value oracle，并进一步生成 verifier、phase oracle、全局碰撞报告和 Grover 逻辑资源报告。

当前 truth-table 编译器是正确性优先的指数级基线，不构成实用量子优势。当前 VQC 模块仍只是潜空间重构先验，也不等价于相干受害模型预言机。下一核心里程碑是实现保持模型结构的可逆定点算术编译器，包括 affine multiply-accumulate、requantization、ReLU/comparator、compute-copy-uncompute 和逐层符号资源界，并以 truth-table 基线对小实例进行穷举等价验证。

在 batch size 1、完整梯度可见且首层带偏置 Linear 直接接收原始输入时，解析攻击已在真实 GIFT-Eval 与 Community Forensics 样本上实现 `within 1e-6 = 100%`；该结论不适用于任意 CNN、聚合梯度或受防御保护的训练过程。
