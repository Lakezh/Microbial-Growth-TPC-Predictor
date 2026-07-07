# 微生物生长温度性能曲线预测工具包

基于物理约束深度学习（UDE/UTPC），通过基因组预测微生物完整**温度性能曲线（TPC）**，
并可通过 FBA 将归一化曲线锚定为绝对生长速率（h⁻¹）。

---

## 工作原理

```
蛋白质组 FASTA（氨基酸序列）
    |
    +---> 425 维蛋白质特征
    |     （氨基酸比例 + 蛋白质组属性 + 二肽频率）
    |              |
    |         OGT MLP（256-128-64）
    |              |
    |         OGT（°C）<-- 硬锚定 Topt
    |
    +---> ESM-2 均值池化嵌入（1280 维）
    |              |
    |       core_model（UDE）
    |       ESM 编码器 + UTPC ODE + 约束残差 MLP
    |              |
    |       归一化 TPC 形状（峰值=1）
    |              |
    |  [可选] CarveMe GEM + COBRApy FBA --> 峰值生长速率（h⁻¹）
    |              |
    绝对 TPC(T) = 归一化形状(T) × 峰值速率
```

---

## 模型与方法

### 第一阶段 — OGT 预测（`OGT_predictor.py`）

最适生长温度（OGT）由**多层感知机（MLP）**预测，训练数据来自
3131 个基因组（2869 细菌 + 262 古菌，GTDB 分类），提取 **425 维蛋白质序列特征**。
所有特征均直接从氨基酸序列计算，只需蛋白质组 FASTA，无需基因组序列、rRNA、
tRNA 或密码子数据。

**特征组成（共 425 维）：**

| 特征组 | 维数 |
|---|---|
| 原始氨基酸比例（20 种氨基酸） | 20 |
| 蛋白质组属性（均值长度、带电/疏水比例、IVYWREL 比值） | 5 |
| 二肽频率（20×20） | 400 |

**架构：** 隐藏层 (256, 128, 64)，ReLU，Adam，自适应学习率，早停。  
**10 折交叉验证（n=3131）：** RMSE 5.17°C | MAE 3.95°C | R² 0.86

### 第二阶段 — TPC 形状预测（`core_model.py` + `TPC_predictor.py`）

归一化 TPC 形状由**通用微分方程（UDE）**模型预测，包含两个组件：

**编码器（`ESMTempEncoder_MLP`）：**  
蛋白质组通过 ESM-2 均值池化嵌入表示（1280 维，模型：`esm2_t33_650M_UR50D`）。
基于 Patch 的 Transformer 编码器将嵌入向量映射到 64 维潜在向量 z，
参数头从中输出 UTPC 物理参数 Pmax 和 E。

**物理约束（UTPC ODE）：**  
预测参数驱动基于 Eppley 动力学的**通用温度性能曲线（UTPC）**：

```
dμ/dT = -(Pmax / E) · exp((T - Topt) / E) · (T - Topt) / E
```

OGT 直接作为 Topt 的**硬锚点**（不加噪声模拟）。

**残差修正（`ResidualMLP`）：**  
小型残差 MLP 修正 ODE 轨迹，通过 `softplus` 门控约束 OGT 以上不再增长。

**训练流程：** 热身（25 轮）→ 交替训练 θ/残差（4 个周期 × 8 轮）→ 联合微调（20 轮）。
损失函数：SmoothL1 数据损失 + 单调性惩罚 + 尾部惩罚。

### 第三阶段 — 绝对锚定（`FBA_anchor_point.py`）

**CarveMe** 从蛋白质组 FASTA 重建基因组规模代谢模型（GEM）。
**COBRApy FBA** 在用户指定培养基下求解，最优生长速率（h⁻¹）将归一化 TPC 锚定到绝对单位：

```
绝对 TPC(T) = 归一化形状(T) × FBA 峰值速率
```

若未提供代谢模型，则输出保持归一化形式（峰值 = 1）。

---

## 文件说明

### `code/core_model.py`
UDE TPC 形状模型的训练脚本。

主要内容：
- `load_data(data_csv)` — 读取含 ESM-2 嵌入列的 TPC 数据集，过滤细菌/古菌，
  补全缺失 OGT，标准化 ESM 嵌入。
- `build_curves(...)` — 按 TPC_id 分组，构建逐曲线数据字典
  （温度数组、归一化形状、嵌入向量、OGT）。
- `train_all(df, esm_cols)` — 完整训练流程：热身 → 交替 → 联合。
  逐轮损失记录到 `Train/core_model_training_log.csv`，保存损失曲线图。
- `save_model(...)` — 保存 `results/core_model_checkpoint.pt` 和
  `results/core_model_scaler.pkl`。
- 神经网络类：`PositionalEncoding`、`ESMTempEncoder_MLP`、`ParamHead`、
  `ResidualMLP`、`UTPC_ODEFunc_Constrained`、`UDEModel_Constrained`。

### `code/TPC_predictor.py`
推理脚本 — 从 ESM-2 嵌入向量和 OGT 值预测归一化 TPC 形状。

主要内容：
- `load_model(checkpoint_path, scaler_path)` — 加载已训练的 UDE 权重和 ESM scaler，
  返回 `model, scaler, meta, device`。
- `predict_shape(model, scaler, meta, device, esm_embedding, ogt_c, temperatures)` —
  预测单个生物体的归一化 TPC。返回 `pred_shape`、`Pmax`、`ToptC`、`E`。
- `predict_from_csv(...)` — 从含 ESM 列和 OGT 列的 CSV 文件批量预测。
- `plot_prediction(result, title, save_path)` — 绘制预测 TPC 曲线。
- 神经网络类（与 `core_model.py` 完全一致，加载权重所需）。

### `code/OGT_predictor.py`
从蛋白质序列特征训练和应用 OGT MLP。

主要内容：
- `train_ogt_mlp(bacteria_csv, archaea_csv)` — 10 折交叉验证后训练最终模型；
  自动从输入 CSV 中选取 425 维蛋白质兼容特征列（去除基因组级与密码子特征）；
  保存 `results/ogt_mlp/mlp.pkl`、`scaler.pkl`、`feature_cols.pkl`、`cv_results.json`。
- `extract_protein_features(protein_fasta)` — 读取蛋白质组 FASTA，计算 425 维特征
  （氨基酸比例、蛋白质组属性、二肽频率）。
- `predict_ogt_from_fasta(fasta_path, model_dir)` — 端到端：
  蛋白质组 FASTA → 425 特征 → OGT。
- `predict_ogt_from_csv(feature_csv, model_dir)` — 从预计算特征 CSV 批量预测 OGT。

### `code/FBA_anchor_point.py`
基因组规模代谢模型重建与 FBA。

主要内容：
- `reconstruct_gem(proteome_fasta, output_xml, universe)` — 调用 CarveMe 构建 SBML GEM。
- `run_fba(gem_path, medium)` — 用 COBRApy 加载 GEM，设置交换反应边界，
  求解 FBA，返回最优生长速率（h⁻¹）。
- `get_peak_growth_rate(fasta_path, medium, temperature_c, gem_path)` — 高层 API：
  自动检测核苷酸/氨基酸 FASTA，可选 Prodigal 调用，CarveMe + FBA 一步完成。

### `examples/example_ecoli.py`
大肠杆菌 K-12 MG1655（中温菌，OGT 37°C）的可运行示例。
使用占位随机嵌入向量；替换为真实嵌入后即可获得有意义的预测。

### `examples/example_thermus.py`
栖热菌 HB8（嗜热菌，OGT 65°C）的可运行示例。

### `examples/example_medium_ecoli.json`
大肠杆菌 iJO1366 的 M9 最小培养基：葡萄糖、O₂、NH₄⁺、磷酸盐、硫酸盐及微量矿物质。
交换反应 ID 遵循 BiGG 命名规范。

### `examples/example_medium_thermus.json`
栖热菌培养基（与大肠杆菌培养基碳/氮/矿物质组成相同）。

### `results/`
已训练的模型文件（已提交到仓库）：
- `core_model_checkpoint.pt` — UDE 编码器/参数头/残差权重 + 元数据（emb_len、
  n_patches、t_mean_k、t_std_k、esm_cols、hyperparams）。
- `core_model_scaler.pkl` — 在 ESM 嵌入上拟合的 `sklearn.StandardScaler`。
- `ogt_mlp/mlp.pkl` — 在 3131 个基因组（425 维蛋白质特征）上训练的最终 OGT MLP。
- `ogt_mlp/scaler.pkl` — 在 425 维蛋白质特征上拟合的 `StandardScaler`。
- `ogt_mlp/feature_cols.pkl` — 425 个特征列名的有序列表（预测时对齐特征顺序）。
- `ogt_mlp/cv_results.json` — 每折的 10 折交叉验证指标。

### `Train/`
训练过程中生成的记录（默认不提交）：
- `core_model_training_log.csv` — 逐轮损失明细（数据 / 正则 / 单调 / 尾部）。
- `core_model_training_loss.png` — 训练损失曲线图。
- `ogt_mlp_cv_log.csv` — 每折 RMSE / MAE / R2。

---

## 分步使用示例：以大肠杆菌 K-12 MG1655 为例

本示例以大肠杆菌 K-12 MG1655 为例，演示如何预测完整 TPC，并可选择通过 FBA 转换为绝对生长速率。

### 第 0 步 — 安装依赖

```bash
pip install -r requirements.txt

# ESM-2 蛋白质组嵌入（用于 TPC 形状预测）：
pip install fair-esm          # 推荐：Facebook 官方库
# 或：pip install transformers

# FBA（可选）：
pip install cobra carveme
```

### 第 1 步 — 下载蛋白质组

从 NCBI RefSeq 下载大肠杆菌 K-12 MG1655 蛋白质组：

```
登录号：GCF_000005845.2
文件：  protein.faa
```

```bash
# 使用 NCBI datasets CLI（https://www.ncbi.nlm.nih.gov/datasets/）：
datasets download genome accession GCF_000005845.2 --include protein
unzip ncbi_dataset.zip
# 蛋白质组 FASTA 路径：ncbi_dataset/data/GCF_000005845.2/protein.faa
```

### 第 2 步 — 从蛋白质组特征预测 OGT

OGT MLP 直接从蛋白质组 FASTA 提取 425 维特征（氨基酸比例、二肽频率、蛋白质组属性）
并预测 OGT，无需基因组序列或基因预测工具：

```bash
python code/OGT_predictor.py predict \
    --fasta ncbi_dataset/data/GCF_000005845.2/protein.faa \
    --model_dir results/ogt_mlp
```

预期输出：
```
Predicted OGT: 36.8 C
```

> 若已知 OGT，可跳过此步，在第 4 步直接通过参数指定：`ogt_c = 37.0`。

### 第 3 步 — 计算 ESM-2 蛋白质组嵌入

用 ESM-2 对所有蛋白质进行嵌入并均值池化（使用与第 2 步相同的 FASTA 文件）：

```python
import esm, torch, numpy as np

model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model_esm.eval()
batch_converter = alphabet.get_batch_converter()

# 读取蛋白质序列
seqs = []
with open("ncbi_dataset/data/GCF_000005845.2/protein.faa") as f:
    h, s = "", ""
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if h and s: seqs.append((h, s))
            h = line[1:].split()[0]; s = ""
        else:
            s += line
    if h and s: seqs.append((h, s))

# 分批嵌入并均值池化
embeddings = []
for i in range(0, len(seqs), 8):
    batch = [(h, s[:1022]) for h, s in seqs[i:i+8]]
    _, _, tokens = batch_converter(batch)
    with torch.no_grad():
        out = model_esm(tokens, repr_layers=[33])
    for j, (_, s) in enumerate(batch):
        embeddings.append(out["representations"][33][j, 1:len(s)+1].mean(0).numpy())

esm_embedding = np.mean(embeddings, axis=0).astype(np.float32)  # shape (1280,)
np.save("ecoli_esm_embedding.npy", esm_embedding)
print(f"ESM 嵌入维度：{esm_embedding.shape}")
```

### 第 4 步 — 预测归一化 TPC 形状

```python
import numpy as np, sys
sys.path.insert(0, "code")
from TPC_predictor import load_model, predict_shape, plot_prediction

# 加载模型
model, scaler, meta, device = load_model()

# 加载嵌入和预测 OGT
esm_embedding = np.load("ecoli_esm_embedding.npy")
ogt_c         = 36.8   # 第 2 步预测值（或直接用已知值 37.0）
temperatures  = np.arange(5, 75, 1, dtype=np.float32)

# 预测
result = predict_shape(model, scaler, meta, device,
                       esm_embedding=esm_embedding,
                       ogt_c=ogt_c,
                       temperatures=temperatures)

print(f"Topt:    {result['ToptC']:.1f} C")
print(f"Pmax:    {result['Pmax']:.4f}")
print(f"E:       {result['E']:.4f}")
print(f"峰值温度：{temperatures[result['pred_shape'].argmax()]:.0f} C")

# 保存结果
import pandas as pd
pd.DataFrame({"temperature_C": result["temperatures"],
              "norm_shape":    result["pred_shape"]}).to_csv("ecoli_tpc.csv", index=False)
plot_prediction(result, title="大肠杆菌 K-12 MG1655", save_path="ecoli_tpc.png")
```

`ecoli_tpc.csv` 前几行示例：

```
temperature_C,norm_shape
5.0,0.012
10.0,0.041
15.0,0.112
...
37.0,1.000
...
55.0,0.023
```

### 第 5 步 — （可选）FBA 绝对生长速率锚定

```python
import sys, json, numpy as np
sys.path.insert(0, "code")
from FBA_anchor_point import get_peak_growth_rate

with open("examples/example_medium_ecoli.json") as f:
    medium = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

# 重建 GEM 并运行 FBA
peak_rate = get_peak_growth_rate(
    fasta_path    = "ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna",
    medium        = medium,
    temperature_c = 37.0,
)
print(f"FBA 峰值生长速率：{peak_rate:.4f} h-1")

# 将归一化曲线转换为绝对生长速率
absolute_tpc = result["pred_shape"] * peak_rate
```

预期输出：
```
FBA 峰值生长速率：0.9821 h-1
```

> **注意：** CarveMe 需要有效的 DIAMOND 数据库及兼容的求解器（CPLEX 或 GLPK），
> 详见 https://carveme.readthedocs.io。

### 第 6 步 — 运行内置示例

以上步骤已在 `examples/example_ecoli.py` 中预置（使用占位嵌入向量）。
替换 `esm_embedding` 变量为真实嵌入后运行：

```bash
python examples/example_ecoli.py
# 输出：examples/output/ecoli_tpc.csv
#        examples/output/ecoli_tpc.png
```

---

## 重新训练模型

### 重新训练 OGT MLP

分别提供细菌和古菌的特征 CSV 文件（包含 `OGT` 列即可，脚本会自动筛选
425 维蛋白质兼容特征，丢弃基因组级及密码子特征）：

```bash
python code/OGT_predictor.py train \
    --bacteria_csv data/calculated_features_bacteria.csv \
    --archaea_csv  data/calculated_features_archaea.csv
```

结果保存到 `results/ogt_mlp/`，训练日志保存到 `Train/ogt_mlp_cv_log.csv`。

### 重新训练核心 TPC 形状模型

准备含 `TPC_id`、`binomial_name`、`temperature`、`mu`、`OGT`、`kingdom`
以及 ESM-2 嵌入列 `esm2_0` … `esm2_1279` 的 TPC 数据集 CSV：

```bash
python code/core_model.py --data data/your_tpc_dataset.csv
```

结果保存到 `results/`，训练日志和损失曲线保存到 `Train/`。

---

## OGT 模型性能

基于 3131 个基因组（2869 细菌 + 262 古菌，GTDB 分类）的 526 维基因组特征，10 折交叉验证结果：

| 评估方式 | RMSE（°C） | MAE（°C） | R² |
|---|---|---|---|
| 10 折 CV 总体 | 5.17 | 3.95 | 0.86 |
| 最优折 | 4.82 | 3.54 | 0.89 |
| 最差折 | 5.76 | 4.34 | 0.81 |

训练集：2869 细菌 + 262 古菌（GTDB 分类），425 维蛋白质序列特征。

---

## 引用

如果您在研究中使用了本工具包，请引用：

- **Hybrid-TPC-Model** — UDE / UTPC 架构来源。
- **OGTfinder** — 526 维基因组特征提取及 OGT MLP 训练方案。
- ESM-2：Lin et al. (2023) *Science* 379, 1123–1130（用于 TPC 形状预测）。
- COBRApy：Ebrahim et al. (2013) *BMC Systems Biology* 7, 74。

---

## 许可证

MIT
