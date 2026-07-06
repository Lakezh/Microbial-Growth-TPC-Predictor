# 微生物生长温度性能曲线预测工具包

基于物理约束深度学习，通过基因组 FASTA 文件预测微生物完整**温度性能曲线（TPC）**，
并可通过 FBA 将归一化曲线锚定为绝对生长速率。

---

## 工作原理

```
基因组 FASTA
    |
    |---[Prodigal 基因预测]---> 蛋白质序列
    |       |
    |       |--> ESM-2 均值池化嵌入向量（1280 维）
    |       |          |
    |       |    +-----+-----------------------------+
    |       |    |   core_model.py（UDE 模型）        |
    |       |    |   ESM编码器 + UTPC ODE 物理约束    |
    |       |    |   + 约束残差 MLP 修正              |
    |       |    +-----+-----------------------------+
    |       |          | 归一化 TPC 曲线形状（峰值 = 1）
    |       |          v
    |       +--> 密码子/氨基酸/二肽特征（526 维）
    |                  |
    |            +-----+----------+
    |            |  OGT_predictor |  sklearn MLP（256-128-64）
    |            +-----+----------+
    |                  | OGT（°C）<-- 锚定 UTPC 中的 Topt
    |
    |---[CarveMe GEM 重建]
    |           |
    |    +-------+----------+
    |    | FBA_anchor_point |  COBRApy FBA（使用指定培养基）
    |    +-------+----------+
    |            | 峰值生长速率（h-1）
    |
    绝对 TPC(T) = 归一化形状(T) × 峰值速率
```

### 第一阶段 —— OGT 预测
`OGT_predictor.py` 提取 526 维基因组特征（rRNA 组成、氨基酸使用频率、密码子使用频率、二肽频率），
使用在 3131 个基因组（2869 细菌 + 262 古菌）上训练的 MLP 预测最适生长温度（OGT）。

10 折交叉验证性能：**RMSE 5.12°C | MAE 3.91°C | R2 0.87**

### 第二阶段 —— TPC 形状预测（核心模型）
`core_model.py` 训练 UDE（通用微分方程）模型：基于 ESM-2 的 Transformer 编码器预测
UTPC 物理参数（Pmax、E），以 Eppley 型 ODE 为物理基础，约束残差 MLP 修正轨迹。
OGT 直接作为 Topt 的硬锚点。

### 第三阶段 —— 绝对锚定（可选）
`FBA_anchor_point.py` 使用 CarveMe 重建基因组规模代谢模型（GEM），
在给定培养基条件下运行 FBA，获取绝对峰值生长速率。

---

## 目录结构

```
Microbial-Growth-TPC-Predictor/
|-- code/
|   |-- core_model.py          UDE TPC 形状模型训练脚本
|   |-- TPC_predictor.py       主入口：FASTA + 培养基 -> 绝对 TPC
|   |-- OGT_predictor.py       OGT MLP：训练与推理
|   |-- FBA_anchor_point.py    CarveMe GEM 重建 + COBRApy FBA
|-- data/                      （空目录 — 请在本地放置训练数据）
|-- results/
|   |-- core_model_checkpoint.pt   UDE 模型权重
|   |-- core_model_scaler.pkl      ESM 嵌入标准化器
|   |-- ogt_mlp/
|       |-- mlp.pkl                训练好的 OGT MLP
|       |-- scaler.pkl             特征标准化器
|       |-- feature_cols.pkl       526 个特征名称列表
|       |-- cv_results.json        10 折交叉验证指标
|-- Train/
|   |-- core_model_training_log.csv    逐轮损失记录
|   |-- core_model_training_loss.png   损失曲线图
|   |-- ogt_mlp_cv_log.csv             逐折交叉验证记录
|-- examples/
|   |-- example_ecoli.py               大肠杆菌 K-12 MG1655（中温菌）
|   |-- example_thermus.py             栖热菌 HB8（嗜热菌）
|   |-- example_medium_ecoli.json      大肠杆菌 M9 最小培养基
|   |-- example_medium_thermus.json    栖热菌培养基
|-- requirements.txt
|-- README.md
|-- READMECN.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

FBA 支持（可选）：

```bash
pip install cobra carveme
# CarveMe 还需要 DIAMOND；详见 https://carveme.readthedocs.io
```

### 2. 准备数据并训练模型

#### 2a. 训练 OGT MLP

```bash
python code/OGT_predictor.py train \
    --bacteria_csv data/calculated_features_bacteria.csv \
    --archaea_csv  data/calculated_features_archaea.csv
```

该脚本运行 10 折交叉验证、打印评估指标、用全量数据训练最终模型，
并将结果保存到 `results/ogt_mlp/`，交叉验证日志保存到 `Train/`。

#### 2b. 训练核心 TPC 形状模型

将含 ESM-2 嵌入列的 TPC CSV 放到 `data/` 后运行：

```bash
python code/core_model.py --data data/11800TPC_1_1_with_medium_group_3_with_OGT.csv
```

训练结果保存到 `results/`，训练日志和损失曲线保存到 `Train/`。

### 3. 从基因组 FASTA 预测 TPC

**仅输出归一化曲线（无 FBA）：**

```bash
python code/TPC_predictor.py \
    --fasta  genome.fna \
    --temp_min 5 --temp_max 80
```

**含 FBA 绝对锚定：**

```bash
python code/TPC_predictor.py \
    --fasta  genome.fna \
    --medium examples/example_medium_ecoli.json \
    --temp_min 5 --temp_max 80
```

**手动指定 OGT（跳过 OGT 预测）：**

```bash
python code/TPC_predictor.py --fasta genome.fna --ogt 37.0
```

完整流程所需的外部工具：
- **Prodigal**（基因预测）—— https://github.com/hyattpd/Prodigal
- **Barrnap**（rRNA 检测）—— https://github.com/tseemann/barrnap
- **ESM-2** —— `pip install fair-esm` 或 `pip install transformers`
- **CarveMe**（GEM 重建，仅 FBA 需要）—— `pip install carveme`

### 4. Python API

```python
import numpy as np
import sys
sys.path.insert(0, "code")

from TPC_predictor import load_model, predict_shape

model, scaler, meta, device = load_model()

# esm_embedding：目标蛋白质组的 ESM-2 均值池化向量，形状 (1280,)
result = predict_shape(
    model, scaler, meta, device,
    esm_embedding = esm_embedding,
    ogt_c         = 37.0,
    temperatures  = np.arange(5, 75, 1),
)
print(f"Topt = {result['ToptC']:.1f} C")
print(f"峰值温度 = {result['temperatures'][result['pred_shape'].argmax()]:.0f} C")
```

---

## 运行示例

```bash
# 大肠杆菌 K-12（中温菌，OGT ~37°C）
python examples/example_ecoli.py

# 栖热菌 HB8（嗜热菌，OGT ~65°C）
python examples/example_thermus.py
```

输出保存到 `examples/output/`。

---

## 输入说明

### 基因组 FASTA
核苷酸基因组 FASTA（`.fna`）或氨基酸蛋白质组 FASTA（`.faa`）。
流程自动检测类型（>80% ACGTN 字符 = 核苷酸）。

### 培养基（FBA 用）
JSON 字典，键为 COBRA 交换反应 ID，值为最大摄取速率（mmol gDW-1 h-1，正值）。
大肠杆菌 M9 最小培养基示例见 `examples/example_medium_ecoli.json`。

### 特征 CSV（OGT 训练用）
每行一个基因组，每列一个特征，必须包含 `OGT` 列。
列名须与 `results/ogt_mlp/feature_cols.pkl` 中存储的名称一致。

---

## OGT 模型性能

| 划分方式          | RMSE（°C） | MAE（°C） | R2   |
|------------------|-----------|----------|------|
| 10 折 CV 总体    | 5.12      | 3.91     | 0.87 |
| 最佳折           | 4.58      | 3.50     | 0.92 |
| 最差折           | 5.75      | 4.31     | 0.80 |

训练集：2869 株细菌 + 262 株古菌（GTDB 分类体系）。

---

## 引用

如果您在研究中使用了本工具包，请引用：

- **Hybrid-TPC-Model** —— UDE / UTPC 架构来源。
- 基因组特征数据与 OGT 标签来源（如 TEMPURA 数据库、Engqvist 2018）。
- COBRApy（FBA 工具）：Ebrahim et al. (2013) *BMC Systems Biology*。
- ESM-2：Lin et al. (2023) *Science*。

---

## 许可证

MIT
