# Q-RECON

面向经典机器学习模型训练数据重构的量子访问理论、相干预言机与可识别性研究。

## 研究目标

本项目不把量子神经网络简单视为经典优化器的替代品，而是从量子信息与量子查询模型出发，研究三个基础问题：

1. 经典模型中的训练数据在什么条件下仍然可被唯一识别？
2. 普通经典 API 与相干量子预言机之间是否存在严格的攻击能力分离？
3. 在预言机实现成本计入总成本后，训练数据重构能否获得可证明的量子查询优势？

## 核心贡献设想

- 建立经典访问、白盒访问与相干量子访问三类威胁模型。
- 给出训练数据不可识别时的量子信息论下界。
- 将量化经典神经网络自动编译为可逆量子预言机。
- 在唯一可识别的结构化候选空间中设计量子重构算法。
- 研究 QNN 的表达能力、可训练性、可恢复性与资源消耗之间的关系。
- 建立覆盖多种模型、数据模态和访问方式的统一基准。

## 文档

- [研究创新与论文路线](docs/RESEARCH_INNOVATION.md)
- [数据集与下载方式](docs/DATASETS.md)
- [本地验证记录](docs/VERIFICATION.md)

## 已实现功能

- GIFT-Eval 时间序列流式加载器；
- TIME 2026 本地数据适配器；
- Community Forensics Small 图像流式加载器；
- CLOFAI/通用 ImageFolder 适配器；
- 时间序列 MLP 与图像 CNN 受害模型；
- 单样本梯度泄漏与梯度反演攻击；
- 直接、经典生成和变分量子三类重构先验；
- 基于完整梯度 Jacobian 秩的局部可识别性分析；
- 重构质量与量子逻辑资源统计；
- YAML 实验配置和轻量单元测试。

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
```

运行真实数据实验：

```bash
qrecon --config configs/time_gifteval_quantum.yaml
qrecon --config configs/image_community_forensics.yaml
```

实验结果写入 `outputs/`，包括 `report.json` 和重构张量
`reconstruction.pt`。

## 当前状态

项目已形成可运行的第一阶段研究原型。当前量子模块是潜空间 VQC
重构先验，不等价于相干受害模型预言机，因此不宣称端到端量子优势。
后续优势结论必须同时核算数据编码、预言机构造、量子电路、shots 和结果读出成本。
