import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ============================================================
# 1. Load datasets
# ============================================================

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

TARGET = "fault_inducing"

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]


# ============================================================
# 2. Train Logistic Regression
# ============================================================

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)


# ============================================================
# 3. Transform test data
# ============================================================

scaler = model.named_steps["scaler"]
logistic = model.named_steps["logistic"]

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 4. Create SHAP LinearExplainer
# ============================================================

explainer = shap.LinearExplainer(
    logistic,
    X_test_scaled,
)

shap_values = explainer.shap_values(X_test_scaled)


# ============================================================
# 5. Global SHAP feature importance
# ============================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

feature_importance = pd.DataFrame(
    {
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap,
    }
)

feature_importance = feature_importance.sort_values(
    "mean_abs_shap",
    ascending=False,
)

print("\n" + "=" * 70)
print("SHAP GLOBAL FEATURE IMPORTANCE")
print("=" * 70)

print("\nTop features:")
print(feature_importance.to_string(index=False))


# ============================================================
# 6. Save feature importance
# ============================================================

feature_importance.to_csv(
    "data/processed/shap_feature_importance.csv",
    index=False,
)

print("\nSaved feature importance to " "data/processed/shap_feature_importance.csv")


# ============================================================
# 7. SHAP summary plot
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_scaled,
    feature_names=X_test.columns,
    show=False,
)

plt.tight_layout()

plt.savefig(
    "data/processed/shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Saved SHAP summary plot to " "data/processed/shap_summary_plot.png")


# ============================================================
# 8. SHAP bar plot
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_scaled,
    feature_names=X_test.columns,
    plot_type="bar",
    show=False,
)

plt.tight_layout()

plt.savefig(
    "data/processed/shap_feature_importance_plot.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Saved SHAP bar plot to " "data/processed/shap_feature_importance_plot.png")


# ============================================================
# 9. Top 10 features
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 SHAP FEATURES")
print("=" * 70)

for i, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:<35} " f"{row['mean_abs_shap']:.6f}")
