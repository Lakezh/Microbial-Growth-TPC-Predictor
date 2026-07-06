# 微生物生长温度性能曲线预测工具包（MGTP）

本工具包可以根据三类输入，为微生物生成完整的**温度性能曲线（TPC）**：

| 输入 | 说明 |
|---|---|
| 基因组特征 | 526维预计算特征（rRNA、氨基酸使用频率、密码子使用频率等） |
| 蛋白质组ESM嵌入 | 基于ESM-2蛋白质语言模型的均值池化向量 |
| 培养基组成 | FBA交换反应上界（可选） |

---

## 工作原理

```
基因组特征（526维）
       |
       v
  +----------+
  |  OGT MLP |  sklearn MLP（256-128-64），在3131个基因组上训练
  +----------+
       | OGT（°C）
       v
  +----------------------------------------------------+
  |  TPC形状  -  PINN/UDE（来自Hybrid-TPC-Model）      |
  |    ESM Transformer -> UTPC参数（Pmax, E）           |
  |    + 约束残差ODE校正                                |
  +----------------------------------------------------+
       | 归一化形状（峰值=1）
       v
  +----------+
  | FBA锚点  |  COBRApy FBA（使用指定培养基）（可选）
  +----------+
       | 峰值生长速率（h-1）
       v
  绝对TPC：growth_rate(T) = shape(T) × FBA峰值速率
```

### 第一阶段 — OGT预测
MLP替代了原始的±5°C噪声模拟器。
交叉验证性能（10折，n=3131）：**RMSE 5.12°C | MAE 3.91°C | R² 0.87**

### 第二阶段 — TPC形状预测
使用**Hybrid-TPC-Model**中预训练的UDE模型，以ESM-2蛋白质组嵌入为输入，
以预测OGT为Topt锚点，输出归一化的TPC曲线形状。

### 第三阶段 — FBA峰值锚点
在预测OGT下，使用给定培养基运行FBA，所得最大生长速率作为TPC峰值的
绝对锚点。如不提供代谢模型，输出保持归一化形式。

---

## 目录结构

```
MGTP/
├── mgtp/                    # Python 包
│   ├── __init__.py
│   ├── ogt_predictor.py     # OGT MLP 包装器
│   ├── tpc_shape.py         # PINN 形状预测器（架构 + 加载器）
│   ├── fba_anchor.py        # COBRApy FBA 包装器
│   └── pipeline.py          # 端到端流程编排
├── train/
│   └── train_ogt.py         # 训练并评估OGT MLP，保存模型文件
├── examples/
│   ├── example_pipeline.py  # 可运行的示例（场景A和B）
│   └── example_medium_ecoli.json
├── models/                  # （不提交到git）— 将训练好的模型放于此处
│   ├── ogt_mlp/             # mlp.pkl  scaler.pkl  feature_cols.pkl
│   └── tpc_pinn/            # checkpoint.pt  esm_scaler.pkl
├── requirements.txt
├── README.md
└── READMECN.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练OGT模型

```bash
python train/train_ogt.py \
    --bacteria_csv /path/to/calculated_features_bacteria.csv \
    --archaea_csv  /path/to/calculated_features_archaea.csv \
    --output_dir   models/ogt_mlp
```

该脚本将：①运行10折交叉验证并打印评估指标，②用全量数据训练最终模型，
③将模型文件保存到 `models/ogt_mlp/`。

### 3. 放置TPC PINN模型文件

从 Hybrid-TPC-Model 复制已训练的PINN检查点和Scaler：

```bash
cp Hybrid-TPC-Model/results/group4_pinn_checkpoint.pt  models/tpc_pinn/checkpoint.pt
cp Hybrid-TPC-Model/results/group4_pinn_scaler.pkl     models/tpc_pinn/esm_scaler.pkl
```

### 4. 使用Python API运行流程

```python
import numpy as np
import pandas as pd
from mgtp import MGTPipeline

# 不使用FBA，输出归一化TPC
pipe = MGTPipeline(
    ogt_model_dir = "models/ogt_mlp",
    tpc_model_dir = "models/tpc_pinn",
)

T, rate, ogt = pipe.predict(
    esm_embedding     = esm_vec,       # ndarray，形状 (嵌入维度,)
    genomic_features  = feature_df,    # 包含526列特征的DataFrame
    temperature_range = np.arange(5, 70, 1),
)
print(f"预测OGT：{ogt:.1f}°C")
print(f"峰值归一化速率：{rate.max():.4f}")
```

### 5. 使用FBA锚点（输出绝对生长速率）

```python
pipe = MGTPipeline(
    ogt_model_dir  = "models/ogt_mlp",
    tpc_model_dir  = "models/tpc_pinn",
    fba_model_path = "models/iJO1366.xml",
)

medium = {
    "EX_glc__D_e": 10.0,   # 葡萄糖
    "EX_o2_e":     20.0,   # 氧气
    "EX_nh4_e":    10.0,   # 铵根离子（氮源）
}

T, rate, ogt = pipe.predict(
    esm_embedding    = esm_vec,
    genomic_features = feature_df,
    medium           = medium,
    temperature_range = np.arange(5, 70, 1),
)
print(f"FBA峰值生长速率：{rate.max():.4f} h-1")
```

---

## 输入规格说明

### 基因组特征（526列）

按与训练数据相同的流程预计算。列名的精确列表存储于
`models/ogt_mlp/feature_cols.pkl`。

| 特征组 | 维数 |
|---|---|
| 基因组大小 | 1 |
| rRNA核苷酸组成 + MFE/len（5S / 16S / 23S） | 15 |
| tRNA核苷酸组成 + MFE/len | 5 |
| 基因组GC含量（归一化） | 1 |
| 蛋白质组氨基酸比例（原始 + GC归一化） | 40 |
| 蛋白质组属性（均值长度、电荷比例） | 4 |
| 密码子使用频率（同义密码子分布） | 64 |
| 二肽频率 | 400 |

### ESM嵌入向量

使用ESM-2（如 `facebook/esm2_t33_650M_UR50D`）对蛋白质组进行均值池化，
维度须与 `models/tpc_pinn/checkpoint.pt` 中保存的 `emb_len` 一致。

### 培养基（FBA用）

字典格式，键为交换反应ID，值为最大摄取速率（mmol gDW⁻¹ h⁻¹）。
大肠杆菌M9基础培养基示例见 `examples/example_medium_ecoli.json`。

---

## OGT模型性能

| 划分方式 | RMSE（°C） | MAE（°C） | R² |
|---|---|---|---|
| 10折交叉验证（总体） | 5.12 | 3.91 | 0.87 |
| 最佳fold | 4.58 | 3.50 | 0.92 |
| 最差fold | 5.75 | 4.31 | 0.80 |

训练集：2869株细菌 + 262株古菌（GTDB分类体系）。

---

## 引用

如果您在研究中使用了MGTP，请引用：

- **Hybrid-TPC-Model** — PINN/UDE架构来源。
- 基因组特征数据和OGT标签来源（如TEMPURA数据库、Engqvist 2018）。
- COBRApy（FBA工具）：Ebrahim et al. (2013) *BMC Systems Biology*。

---

## 许可证

MIT
