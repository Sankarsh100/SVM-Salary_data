# Salary Classification with Support-Vector Machines (SVM)

Predict whether an individual earns **`<=50K`** or **`>50K`** per year using the classic U.S. Census–style salary dataset.  
This repo contains an end-to-end, **reproducible** machine-learning pipeline built with Python and scikit-learn.

---

## 📈 Project Highlights
| Stage | Key Steps |
|-------|-----------|
| **1. Data Prep** | • Inspect & clean 32 k rows  • Impute / drop nulls  • Label-encode 9 categoricals |
| **2. EDA** | Correlation heat-maps & box-plots reveal strong impact of education, capital-gain & hours-per-week |
| **3. Modeling** | Compare **Linear**, **Poly (d=3)** & **RBF** kernels via `GridSearchCV` |
| **4. Evaluation** | Best RBF model → **~85 % accuracy / 0.88 F1** on held-out test set |
| **5. Explainability** | Permutation Importance & SHAP identify top drivers |
| **6. Packaging** | Single-click notebook; requirements file; results saved to `/output` |

---

## 🗂 Repo Structure
