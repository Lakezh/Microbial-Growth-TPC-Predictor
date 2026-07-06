# Microbial Growth TPC Predictor (MGTP)

MLP-based models for predicting microbial Optimal Growth Temperature (OGT) and temperature-dependent kinetics.

## Structure

- `Chapter2/OGT/` - OGT prediction from genomic features (bacteria + archaea)

## Models

### OGT Predictor
- Input: 526 genomic features (rRNA composition, amino acid usage, codon usage, dipeptide frequencies)
- Architecture: MLP 256-128-64, ReLU, Adam
- Evaluation: 10-fold cross-validation
- Results: RMSE=5.12, MAE=3.91, R2=0.87 (n=3131)
