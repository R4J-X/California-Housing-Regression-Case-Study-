# California Housing Price Prediction — Regression Case Study

Predicting the median house value of a California census block group (1990 US
Census, 20,640 records, 8 features) and comparing a linear baseline with two
non-linear ensemble models.

**Author:** Raj Singh Mall — Roll No. AP24110010967

## Dataset
`sklearn.datasets.fetch_california_housing()` — 20,640 block groups, 8 numerical
features (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude,
Longitude). Target: median house value in units of $100,000 (capped at $500,001).
No missing values.

## Method
- 80/20 random train–test split (`random_state=42`): 16,512 / 4,128
- **Linear Regression** (baseline, features standardised)
- **Random Forest** (100 trees, `n_jobs=-1`)
- **Gradient Boosting** (200 trees, learning rate 0.1, depth 4)
- Metrics: MAE, RMSE, R² on the test set + 5-fold cross-validated R²

## Results

Produced by running `california_housing_case_study.py` (scikit-learn 1.9.1);
raw output is saved in `outputs/`.

| Model | MAE ($) | RMSE ($) | Test R² | 5-fold CV R² | Train time (s) |
|---|---|---|---|---|---|
| Linear Regression | 53,320 | 74,558 | 0.576 | 0.601 ± 0.017 | 0.02 |
| Random Forest | 32,773 | 50,599 | 0.805 | 0.810 ± 0.006 | 8.41 |
| Gradient Boosting * | 32,740 | 48,760 | 0.819 | 0.825 ± 0.009 | 8.02 |

\* best model — 34.6% lower RMSE than the linear baseline.
(Train times depend on the machine used.)

MedInc is the most important feature (importance 0.525), followed by AveOccup
(0.138); Latitude (0.089) and Longitude (0.088) are next, supporting the base
paper's point that location matters most.

![Predictions and feature importance](outputs/fig1_predictions_and_importance.png)

## Repository contents
| File | Description |
|---|---|
| `california_housing_case_study.py` | Full code |
| `requirements.txt` | Python dependencies |
| `outputs/model_metrics.csv` | Table 1 metrics |
| `outputs/metrics.json` | Metrics + run settings |
| `outputs/fig1_predictions_and_importance.png` | Figure 1 |
| `outputs/test_predictions_sorted.csv` | Test predictions sorted by error |
| `outputs/console_output.txt` | Full console log of the run |

## Limitations
Random split (neighbouring blocks leak between train and test), target capped at
$500,001, 1990 data, no hyperparameter tuning, block-level (not house-level) features.

## How to run
```bash
pip install -r requirements.txt
python california_housing_case_study.py
```
Outputs are written to `outputs/` (metrics CSV/JSON, figure, sorted predictions).

## References
1. R. K. Pace, R. Barry, "Sparse spatial autoregressions," Stat. Probab. Lett., 33(3), 291–297, 1997.
2. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," JMLR, 12, 2825–2830, 2011.
3. L. Breiman, "Random forests," Machine Learning, 45(1), 5–32, 2001.
4. J. H. Friedman, "Greedy function approximation: A gradient boosting machine," Ann. Stat., 29(5), 1189–1232, 2001.
