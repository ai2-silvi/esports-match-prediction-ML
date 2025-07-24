# 🎮 CS:GO Map-Winner Prediction – Coursework Project

> **Group assignment uploaded individually**  
> Authors: Zacharias Johansson, Albatool Sassi, Afia Ibnat Silvi 
> University Machine-Learning Course (Spring 2025)

---

## 📌 What this repo contains

| File | Purpose |
|------|---------|
| [`finished-proposal.pdf`](finished-proposal.pdf) | **Project proposal** – motivation, datasets, methodology and evaluation plan. |
| [`vertopal.com_main.pdf`](vertopal.com_main.pdf) | **EDA + modelling notebook** – data cleaning, feature engineering, model training & results. |

---

## 🎯 Goal

**Can we predict the winner of a Counter-Strike map *before* the first round starts?**  
Using **175 k** historical matches and **pre-match** features only, we built ML models that achieve **0.92 ROC-AUC**.

---

## 📊 Dataset snapshot

- **Rows**  : 175 594 map-level observations  
- **Target** : `map_winner` (1 = Team-1 wins, 2 = Team-2 wins)  
- **Features**: team win-rates, map-specific stats, player ADR, HLTV ranks, pick & side info.

---

## 🧪 Models benchmarked

| Model | CV ROC-AUC | CV Log-loss |
|-------|-----------|-------------|
| **LightGBM** | 0.922 | 0.353 |
| **CatBoost** | 0.923 | 0.350 |
| Logistic Regression | 0.911 | 0.376 |

All evaluations use **GroupKFold** stratified by `match_id` to avoid leakage.

---

## 🔍 Key findings

- **Map + side bias** (Inferno, Train, etc.) is a strong predictor.  
- **ADR difference** between rosters is the single most important feature.  
- No severe class imbalance (≈ 53 % Team-1 wins) – class-weighting sufficient.  
- **No post-match leakage** – dropped `round_diff`, pistol winners, etc.

---

## 🚀 Quick start

```bash
# 1. clone repo
git clone https://github.com/ai2-silvi/esports-match-prediction-ML.git
cd esports-match-prediction-ML

# 2. install deps
pip install -r requirements.txt

# 3. train LightGBM
python src/train_lightgbm.py
