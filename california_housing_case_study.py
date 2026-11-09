"""
California Housing Regression Case Study
========================================
Predicting median block-group house values from 8 census features (1990 US Census)
and comparing a linear baseline with two non-linear ensembles.

Author : Raj Singh Mall
Roll No: AP24110010967

Run : python california_housing_case_study.py
Out : ./outputs/  (metrics tables, JSON, figures, example predictions)
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
pd.set_option("display.width", 140)

print("scikit-learn %s | numpy %s | pandas %s"
      % (sklearn.__version__, np.__version__, pd.__version__))

data = fetch_california_housing(as_frame=True)
X = data.data.copy()
y = data.target.copy()
feature_names = list(X.columns)

print("\nRecords:", X.shape[0], "| Features:", X.shape[1])
print("Feature names:", feature_names)
print("\nMissing values per column:\n", X.isna().sum().to_string())
print("\nTarget ($100,000 units): min %.3f  max %.3f  mean %.3f"
      % (y.min(), y.max(), y.mean()))
n_capped = int((y == y.max()).sum())
print("Capped blocks (value = $500,001): %d (%.1f%%)"
      % (n_capped, 100 * n_capped / len(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("\nTraining examples: %d | Test examples: %d" % (len(X_train), len(X_test)))

models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        random_state=RANDOM_STATE),
}

results, predictions = [], {}
for name, model in models.items():
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    results.append({
        "Model": name,
        "MAE ($)": mean_absolute_error(y_test, y_pred) * 1e5,
        "RMSE ($)": np.sqrt(mean_squared_error(y_test, y_pred)) * 1e5,
        "Test R2": r2_score(y_test, y_pred),
        "Train time (s)": train_time,
    })
    print("{:<20s} MAE {:9,.0f} | RMSE {:9,.0f} | R2 {:.3f} | {:.2f}s".format(
        name, results[-1]["MAE ($)"], results[-1]["RMSE ($)"],
        results[-1]["Test R2"], train_time))

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for row, (name, model) in zip(results, models.items()):
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    row["CV R2 mean"], row["CV R2 sd"] = scores.mean(), scores.std()
    print("%-20s CV R2 %.3f +/- %.3f" % (name, scores.mean(), scores.std()))

results_df = pd.DataFrame(results)
best_name = results_df.loc[results_df["RMSE ($)"].idxmin(), "Model"]
results_df["Best"] = np.where(results_df["Model"] == best_name, "*", "")
results_df = results_df[["Model", "MAE ($)", "RMSE ($)", "Test R2",
                         "CV R2 mean", "CV R2 sd", "Train time (s)", "Best"]]

print("\nTable 1 - performance on the held-out test set (20% of data)")
print(results_df.to_string(index=False, float_format="{:,.3f}".format))
results_df.to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)

idx = results_df.set_index("Model")
rmse_lr = idx.loc["Linear Regression", "RMSE ($)"]
rmse_best = idx.loc[best_name, "RMSE ($)"]
print("\nBest model: %s" % best_name)
print("RMSE reduction vs Linear Regression: %.1f%%"
      % (100 * (1 - rmse_best / rmse_lr)))

y_pred_best = predictions[best_name]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(y_test.values * 100, y_pred_best * 100,
           s=6, alpha=0.35, edgecolors="none")
lims = [0, 520]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual value ($ thousands)")
ax.set_ylabel("Predicted value ($ thousands)")
ax.set_title("%s: predicted vs actual" % best_name)
ax.legend(loc="upper left")

ax = axes[1]
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
ax.barh(imp.index, imp.values, color="#1f77b4")
ax.set_xlabel("Importance (impurity-based)")
ax.set_title("Random Forest feature importance")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_predictions_and_importance.png"), dpi=150)
plt.close(fig)

print("\nRandom Forest feature importances (descending):")
print(imp.sort_values(ascending=False).round(3).to_string())

cmp_df = X_test.copy()
cmp_df["Actual ($)"] = y_test.values * 1e5
cmp_df["Predicted ($)"] = y_pred_best * 1e5
cmp_df["Abs Error ($)"] = (cmp_df["Actual ($)"] - cmp_df["Predicted ($)"]).abs()
cmp_df = cmp_df.sort_values("Abs Error ($)")

print("\nTable 2a - smallest errors:")
print(cmp_df.head(3).round(2).to_string())
print("\nTable 2b - largest errors:")
print(cmp_df.tail(3).round(2).to_string())
cmp_df.to_csv(os.path.join(OUT_DIR, "test_predictions_sorted.csv"))


def sse(v):
    v = np.asarray(v, dtype=float)
    return float(((v - v.mean()) ** 2).sum())

income = np.array([2, 3, 6, 8], dtype=float)
value = np.array([1.0, 1.4, 3.0, 3.8], dtype=float)
left, right = value[income < 4.5], value[income >= 4.5]

print("\nWorked example - decision-tree split")
print("SSE before split : %.2f" % sse(value))
print("Left  group {1.0, 1.4}: mean %.2f, SSE %.2f" % (left.mean(), sse(left)))
print("Right group {3.0, 3.8}: mean %.2f, SSE %.2f" % (right.mean(), sse(right)))
print("SSE after split  : %.2f (reduction %.2f)"
      % (sse(left) + sse(right), sse(value) - sse(left) - sse(right)))
print("Prediction for a block with MedInc = 7 (x $10,000): $%.0f"
      % (right.mean() * 1e5))

metrics = {
    "sklearn_version": sklearn.__version__,
    "n_records": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_capped_blocks": n_capped,
    "best_model": best_name,
    "models": results_df.drop(columns=["Best"]).to_dict(orient="records"),
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\nAll outputs written to:", os.path.abspath(OUT_DIR))
