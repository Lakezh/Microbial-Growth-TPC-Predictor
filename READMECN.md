# 微生物生长温度性能曲线预测工具包

基于物理约束深度学习（UDE/UTPC），通过基因组预测微生物完整**温度性能曲线（TPC）**，
并可通过 FBA 将归一化曲线锚定为绝对生长速率（h⁻¹）。

---

## 工作原理

```
基因组 FASTA
    |
    +---[Barrnap + Prodigal]---> rRNA / tRNA / ORF 序列
    |           |
    |           |--> 526 维基因组特征
    |           |    （基因组大小 + rRNA/tRNA 组成 + GC +
    |           |     氨基酸比例 + 密码子使用 + 二肽频率）
    |           |
    |      OGT_predictor（MLP 256-128-64）
    |           |
    |      OGT（°C）
    |           |
    +---[Prodigal]---> 蛋白质序列
    |       |
    |       |--> ESM-2 均值池化嵌入（1280 维）
    |                  |
    |            core_model（UDE）
    |       ESM 编码器 + UTPC ODE + 约束残差 MLP
    |                  |
    |   OGT <-- 硬锚定 Topt    归一化 TPC 形状（峰值=1）
    |                  |
    +---[CarveMe GEM 重建]
    |           |
    |    +-------+----------+
    |    | FBA_anchor_point |  COBRApy FBA（使用用户指定培养基）
    |    +-------+----------+
    |            | 峰值生长速率（h⁻¹）
    |
    绝对 TPC(T) = 归一化形状(T) × 峰值速率
```

---

## 模型与方法

### 第一阶段 — OGT 预测（`OGT_predictor.py`）

最适生长温度（OGT）由**多层感知机（MLP）**预测，训练数据来自
3131 个基因组（2869 细菌 + 262 古菌，GTDB 分类），提取 526 维基因组序列特征。

**特征组成（共 526 维）：**

| 特征组 | 维数 |
|---|---|
| 基因组大小 | 1 |
| rRNA 核苷酸比例 + MFE/len（5S / 16S / 23S） | 15 |
| tRNA 核苷酸比例 + MFE/len | 5 |
| 基因组 GC 含量 | 1 |
| 蛋白质组氨基酸比例（GC 归一化 + 原始） | 40 |
| 蛋白质组属性（均值长度、电荷比例、热稳定性指数） | 5 |
| ORF 密码子使用频率（59 个密码子，排除终止/M/W） | 59 |
| 二肽频率（20×20） | 400 |

**架构：** 隐藏层 (256, 128, 64)，ReLU，Adam，自适应学习率，早停。  
**10 折交叉验证（n=3131）：** RMSE 5.12°C | MAE 3.91°C | R² 0.87

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
从基因组序列特征训练和应用 OGT MLP（OGTfinder 方法）。

主要内容：
- `train_ogt_mlp(bacteria_csv, archaea_csv)` — 10 折交叉验证后训练最终模型；
  CSV 需含 526 维基因组特征列及 `OGT` 列；保存 `results/ogt_mlp/mlp.pkl`、
  `scaler.pkl`、`feature_cols.pkl`、`cv_results.json`。
- `extract_genomic_features(genome_fasta, tmp_dir)` — 运行 Barrnap + Prodigal，
  提取 526 维基因组特征向量（rRNA/tRNA 核苷酸比例、GC 含量、氨基酸组成、密码子使用、二肽频率）。
- `predict_ogt_from_fasta(fasta_path, model_dir, tmp_dir)` — 端到端：
  FASTA → Barrnap/Prodigal → 526 特征 → OGT。
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
- `ogt_mlp/mlp.pkl` — 在 3131 个基因组（526 维特征）上训练的最终 OGT MLP。
- `ogt_mlp/scaler.pkl` — 在 526 维基因组特征上拟合的 `StandardScaler`。
- `ogt_mlp/feature_cols.pkl` — 526 个特征列名的有序列表（预测时对齐特征顺序）。
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

# 基因组特征提取工具（需要 Linux/Mac 环境或 WSL）：
# Prodigal: https://github.com/hyattpd/Prodigal
# Barrnap:  https://github.com/tseemann/barrnap
# ViennaRNA（可选，用于 rRNA MFE 特征）: https://www.tbi.univie.ac.at/RNA/

# FBA（可选）：
pip install cobra carveme
```

### 第 1 步 — 下载基因组

从 NCBI RefSeq 下载大肠杆菌 K-12 MG1655 基因组：

```
登录号：GCF_000005845.2
文件：  GCF_000005845.2_ASM584v2_genomic.fna
```

```bash
# 使用 NCBI datasets CLI（https://www.ncbi.nlm.nih.gov/datasets/）：
datasets download genome accession GCF_000005845.2 --include genome
unzip ncbi_dataset.zip
# 基因组 FASTA 路径：ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna
```

### 第 2 步 — 从基因组序列特征预测 OGT

OGT MLP 使用 Barrnap + Prodigal 从基因组 FASTA 中提取 526 维序列特征进行预测，
与 TPC 形状预测模块完全独立：

```bash
python code/OGT_predictor.py predict \
    --fasta ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna \
    --model_dir results/ogt_mlp
```

预期输出：
```
[OGT] Running Barrnap to extract rRNA ...
[OGT] Running Prodigal to call ORFs ...
[OGT] Extracted 526 genomic features.
Predicted OGT: 36.8 C
```

> 若 Barrnap 或 Prodigal 不可用，可在第 4 步直接通过参数指定已知 OGT：`ogt_c = 37.0`。

若已预计算特征 CSV，也可直接从特征文件批量预测：

```bash
python code/OGT_predictor.py predict_csv \
    --feature_csv my_features.csv \
    --model_dir results/ogt_mlp
```

### 第 3 步 — 计算 ESM-2 蛋白质组嵌入

用 Prodigal 提取蛋白质序列：

```bash
prodigal \
    -i ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna \
    -a ecoli_proteins.faa \
    -p single -q
```

用 ESM-2 对所有蛋白质进行嵌入并均值池化：

```python
import esm, torch, numpy as np

model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model_esm.eval()
batch_converter = alphabet.get_batch_converter()

# 读取蛋白质序列
seqs = []
with open("ecoli_proteins.faa") as f:
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

准备包含 526 维基因组特征列和 `OGT` 列的 CSV 文件（列名需与
`results/ogt_mlp/feature_cols.pkl` 一致）。可分别提供细菌和古菌 CSV：

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
| 10 折 CV 总体 | 5.12 | 3.91 | 0.87 |
| 最优折 | 4.58 | 3.50 | 0.92 |
| 最差折 | 5.75 | 4.31 | 0.80 |

训练集：2869 细菌 + 262 古菌（GTDB 分类），526 维基因组序列特征。

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
