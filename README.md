
# Multiple Disease Data — EDA & ML Notebook

A clean, portfolio-ready notebook for **multi-disease** data analysis: **EDA → preprocessing → modeling → evaluation**.  
Works as a template for health datasets (heart disease, diabetes, stroke, etc.).

> Main file: **`MULTIPLE DISEASE DATA.ipynb`**

---

## 🎯 Objectives
- Explore distributions, correlations and class imbalance
- Clean / preprocess: missing values, encoding, scaling
- Build baseline & tuned models (e.g., Logistic Regression, Random Forest, XGBoost*)
- Evaluate with **accuracy, precision, recall, F1, ROC-AUC**, confusion matrix
- Export clean features / metrics for reporting

> \* XGBoost is optional; install only if you plan to use it.

---

## 🧱 Stack
Python · pandas · numpy · scikit-learn · matplotlib · seaborn · imbalanced-learn *(optional)* · xgboost *(optional)*

---

## 🚀 Quickstart
```bash
# 1) Create environment
conda create -n mdd python=3.10 -y
conda activate mdd

# 2) Install deps
pip install -r requirements.txt

# 3) Launch notebook
jupyter notebook "MULTIPLE DISEASE DATA.ipynb"
```

> If your dataset is large/private, place it in `data/` (ignored by git) and update the path inside the notebook.

---

## 📁 Repo Structure
```
.
├─ MULTIPLE DISEASE DATA.ipynb           # main analysis notebook
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ LICENSE
├─ data/                                 # put CSVs here (ignored)
└─ assets/                               # images for README (optional)
```

---

## 🔎 What the notebook does
1. **EDA:** shape, dtypes, missing, target balance, histograms, boxplots, correlation heatmap
2. **Preprocessing:** imputation, categorical encoding, scaling (Standard/MinMax), train/val/test split
3. **Models:** Logistic Regression (baseline), Random Forest; optional XGBoost
4. **Evaluation:** confusion matrix, ROC/PR curves, k-fold cross‑val scores
5. **Imbalance (optional):** class weights or SMOTE (imbalanced-learn)

---

## 🧹 Tips
- Clear outputs before committing to GitHub:
  ```bash
  jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "MULTIPLE DISEASE DATA.ipynb"
  ```
- Save charts to `assets/` for the README if needed.

---

## 📝 License
MIT — free to use with attribution.
