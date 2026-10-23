# Project EcoDrive – Summary Report
**Pre-owned EV valuation (regression) and demand prediction (classification)**

## 1. Data
No raw feed was supplied, so a synthetic dataset of 620 EV listings was generated (`ecodrive_raw_dataset.csv`). It reproduces the issues in the brief: 2–6 % missing values per column, 6 odometer readings of 9,999,999 km, 8 inflated prices (3–6× normal), 3 charging-speed sensor glitches, text categoricals and mixed scales. Replace Section 0 of the notebook with `pd.read_csv(...)` to run on real data.

## 2. Pipeline
| Module | What was done |
|---|---|
| 1 Cleaning | Null audit; median imputation (numeric), mode (categorical). 12 rows with missing *target price* were dropped rather than imputed (imputing a label teaches the model our guess). IQR (1.5×) outlier detection, values capped to bounds. |
| 2 Features | Vehicle_Age, Energy_Efficiency_Index = (Range/kWh)/(1+Age), Battery_Degradation_Per_Year, Km_Per_Year, Effective_Range_km; One-Hot encoding of Brand/Chemistry/Region; StandardScaler on continuous features. |
| 3 Selection/validation | RFE (12 features kept) using Logistic/Linear Regression; Stratified 5-fold CV (on `High_Demand` for classification, on price quartiles for regression). Scaling and RFE sit inside each CV pipeline, so there is no leakage. |
| 4 Baselines/tuning | Logistic/Linear Regression, KNN, SVM/SVR with default-style settings; RandomizedSearchCV (6 iterations, 3-fold inner CV) for the ensembles. |
| 5 Ensembles | Random Forest (bagging) and AdaBoost (boosting) with **nested CV**, so the reported scores are unbiased. Final models checked on a 20 % hold-out set. |

## 3. Results (5-fold CV, mean)
**Classification – High_Demand**
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.732 | 0.633 | 0.817 |
| KNN | 0.701 | 0.595 | 0.745 |
| SVM (RBF) | 0.729 | 0.600 | 0.793 |
| Random Forest | 0.727 | 0.612 | 0.782 |
| AdaBoost | 0.704 | 0.597 | 0.778 |

**Regression – Resale_Price_USD**
| Model | R² | RMSE (USD) | MAE (USD) |
|---|---|---|---|
| Linear Regression | 0.891 | 3,897 | 2,444 |
| KNN | 0.864 | 4,389 | 2,892 |
| SVR (RBF) | **0.905** | **3,610** | **2,138** |
| Random Forest | 0.861 | 4,444 | 2,992 |
| AdaBoost | 0.860 | 4,463 | 2,919 |

Hold-out check: SVR R² = 0.949, RMSE about $2,743, MAE about $2,014. The classifier reached accuracy 0.70 on the hold-out set, with fast-sale recall of only 0.43.

## 4. Recommendation
- **Demand classifier:** Logistic Regression (best F1 and ROC-AUC, and simple to explain). To catch more fast sellers, lower the decision threshold, since recall for the "fast" class is modest.
- **Price model:** SVR (RBF), typically within about $2–3.6k of the true price.

## 5. Honest caveats
- **Ensembles did not win here.** The brief anticipated that ensembles would be best, but on this dataset the smoother models (Linear/SVR/Logistic) edged them out, as the synthetic price follows a smooth relationship. Differences between models are mostly within one standard deviation, so treat the ranking as indicative.
- All findings are based on **synthetic data**; re-run on real dealership data before drawing business conclusions.
- Capping IQR outliers also caps some genuinely expensive vehicles; the threshold can be relaxed (e.g. 3×IQR).
- Tuning budget was kept small (6 iterations) for runtime; increase `N_ITER` for a deeper search.

## 6. Files
`Project_EcoDrive.ipynb` (executed notebook), `ecodrive_raw_dataset.csv`, `ecodrive_cleaned_features.csv`, `classification_results.csv`, `regression_results.csv`, `figures/` (10 plots).
