# 微生物生长温度性能曲线预测工具包

基于物理约束深度学习（UDE/UTPC），输入蛋白质组 FASTA、温度区间和培养基信息，
预测微生物完整**温度性能曲线（TPC）**，并通过 FBA 将归一化曲线锚定为绝对生长速率（h⁻¹）。

---

## 工作原理

**三类输入 → 完整 TPC：**

| 输入 | 说明 |
|---|---|
| 蛋白质组 FASTA | 氨基酸序列（也可输入基因组 FASTA，自动调用 Prodigal 转换） |
| 温度区间 | 例如 5–80°C，步长 1°C |
| 培养基 JSON | 交换反应 ID + 摄取速率，用于 FBA 绝对定量 |

```
蛋白质组 FASTA  （基因组 → Prodigal → 蛋白质，若输入核苷酸序列）
    |
    +---> 425 维蛋白质特征 --> OGT MLP --> OGT（°C）  [硬锚定 Topt]
    |     （氨基酸比例 + 二肽频率 + 蛋白质组属性）
    |
    +---> ESM-2 均值池化嵌入（1280 维）
    |              |
    |       core_model（UDE/UTPC）
    |       ESM 编码器 + UTPC ODE + 约束残差 MLP
    |              |
    |       归一化 TPC 形状（峰值=1）在用户指定温度区间上
    |              |
    |  培养基 JSON --> CarveMe GEM + COBRApy FBA --> 峰值生长速率（h⁻¹）
    |              |
    绝对 TPC(T) = 归一化形状(T) × 峰值速率   [h⁻¹，逐温度点]
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

## 使用示例：以大肠杆菌 K-12 MG1655 为例

生成完整绝对量纲 TPC 需要三类输入：

| # | 输入 | 说明 |
|---|---|---|
| 1 | **蛋白质组 FASTA** | 氨基酸序列（也可输入基因组 FASTA，自动调用 Prodigal 转换） |
| 2 | **温度区间** | 最小值 / 最大值 / 步长（°C） |
| 3 | **培养基 JSON** | 交换反应 ID + 摄取速率 |

### 第 0 步 — 安装依赖

```bash
pip install -r requirements.txt

# ESM-2（用于 TPC 形状预测）：
pip install fair-esm          # 推荐：Facebook 官方库
# 或：pip install transformers

# FBA（用于绝对生长速率定量）：
pip install cobra carveme
```

### 第 1 步 — 准备三类输入

**1a. 下载蛋白质组**（NCBI RefSeq，登录号 `GCF_000005845.2`）：

```bash
# 使用 NCBI datasets CLI（https://www.ncbi.nlm.nih.gov/datasets/）：
datasets download genome accession GCF_000005845.2 --include protein
unzip ncbi_dataset.zip
# 蛋白质组 FASTA：ncbi_dataset/data/GCF_000005845.2/protein.faa
```

> 若只有基因组 FASTA，直接传入即可，流程会自动调用 Prodigal 提取蛋白质序列。

**1b. 确定温度区间**，例如 5–80°C，步长 1°C（通过 `--temp_min/max/step` 参数设置）。

**1c. 准备培养基 JSON**，格式为 BiGG 交换反应 ID → 最大摄取速率（mmol gDW⁻¹ h⁻¹）。
仓库已提供大肠杆菌 M9 最小培养基示例（`examples/example_medium_ecoli.json`）：

```json
{
  "EX_glc__D_e": 10.0,
  "EX_o2_e":     20.0,
  "EX_nh4_e":    10.0,
  "EX_pi_e":     10.0,
  "EX_so4_e":    10.0
}
```

### 第 2 步 — 一键运行完整流程

```bash
python code/TPC_predictor.py \
    --fasta   ncbi_dataset/data/GCF_000005845.2/protein.faa \
    --medium  examples/example_medium_ecoli.json \
    --temp_min 5 --temp_max 80 --temp_step 1 \
    --output  ecoli_tpc.csv
```

脚本自动完成：
1. 提取 425 维蛋白质特征 → OGT MLP 预测 OGT
2. ESM-2 均值池化嵌入 → UDE/UTPC 预测归一化 TPC 形状
3. CarveMe 重建 GEM + COBRApy FBA → 峰值生长速率
4. 形状 × 峰值速率 → 绝对 TPC 写入 `ecoli_tpc.csv`

预期输出：
```
[OGT] Predicted OGT = 36.8 C
[ESM] Embedding 4321 proteins with ESM-2 ...
[FBA] Growth rate = 0.9821 h-1

Results saved to: ecoli_tpc.csv
OGT used:   36.8 C
UTPC Pmax:  3.2415
UTPC E:     8.7632
Plot saved to: ecoli_tpc.png
```

`ecoli_tpc.csv`：

```
temperature_C,norm_shape,abs_growth_rate_per_h
5.0,0.012,0.0118
...
37.0,1.000,0.9821
...
80.0,0.001,0.0010
```

> **仅输出归一化曲线：** 去掉 `--medium` 参数（无需 CarveMe）。  
> **已知 OGT：** 添加 `--ogt 37.0` 跳过 OGT 预测步骤。

### 第 3 步 — Python API（高级用法）

```python
import numpy as np, json, sys
sys.path.insert(0, "code")
from TPC_predictor import run_pipeline

with open("examples/example_medium_ecoli.json") as f:
    medium = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

result = run_pipeline(
    fasta_path   = "ncbi_dataset/data/GCF_000005845.2/protein.faa",
    temperatures = np.arange(5, 81, 1, dtype=np.float32),
    medium       = medium,       # 省略则仅输出归一化曲线
    # ogt_c      = 37.0,         # 取消注释可直接指定 OGT
)

print(f"OGT：{result['ogt_c']:.1f} C")
print(f"峰值生长速率：{result['abs_growth_rate'].max():.4f} h-1")

import pandas as pd
pd.DataFrame({
    "temperature_C":         result["temperatures"],
    "norm_shape":            result["norm_shape"],
    "abs_growth_rate_per_h": result["abs_growth_rate"],
}).to_csv("ecoli_tpc.csv", index=False)
```

> **注意：** CarveMe 需要有效的 DIAMOND 数据库及兼容的求解器（CPLEX 或 GLPK），
> 详见 https://carveme.readthedocs.io。

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

基于 3131 个基因组（2869 细菌 + 262 古菌，GTDB 分类）的 425 维蛋白质序列特征，10 折交叉验证结果：

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

学术非商业许可证（Academic Non-Commercial License）——仅限非商业学术与科研用途免费使用，需保留署名；商业用途需事先获得书面许可。完整条款见 [LICENSE](LICENSE)。
