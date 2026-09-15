# HDER

本仓库用于支撑论文《HDER：基于跨层注意力与图推理的长文档关系抽取研究》，包含公开代码、实验结果和数据准备说明。

## 目录

- `Code/`：HDER训练、评估、消融实验及辅助脚本。
- `Experimental_Results/`：论文主要实验结果数据。
- `Data/`：数据准备说明与实验协议。

## 实验结果

实验结果按来源和实验类型分开组织：

- `Main/`：DocRED、Re-DocRED主实验结果及文献基线。
- `Ablation/`：论文表5消融实验结果。
- `Grouped_Analysis/`：分组分析与层次深度实验结果。
- `Efficiency/`：论文表6模型复杂度与运行效率结果。
- `Error_Analysis/`：论文表7误差分析结果。

详细说明见 `Experimental_Results/README.md`。

## 数据说明

DocRED 和 Re-DocRED 为第三方公开基准数据集，本仓库不重复分发，下载地址与本地文件说明见 `Data/README.md`。

`Code/auxiliary_graph_data/` 仅保留目录占位文件，不包含本地辅助图统计文件。

## 运行

进入 `Code/` 目录后可运行：

```bash
bash train.sh full
```

测试阶段默认加载验证集指标选择得到的 `best.ckpt`。其他消融实验命令见 `train.sh`。
