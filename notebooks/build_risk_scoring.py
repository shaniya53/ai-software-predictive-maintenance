import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ============================================================
# 1. Load data
# ============================================================

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

TARGET = "fault_inducing"

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_test = test.drop(columns=[TARGET])


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
# 3. Predict fault probability
# ============================================================

probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# 4. Convert probability into risk level
# ============================================================


def assign_risk(probability):
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    else:
        return "High"


risk_levels = [assign_risk(p) for p in probabilities]


# ============================================================
# 5. Create risk-scored dataset
# ============================================================

risk_data = X_test.copy()

risk_data["fault_probability"] = probabilities
risk_data["risk_percentage"] = probabilities * 100
risk_data["risk_level"] = risk_levels


# ============================================================
# 6. Save results
# ============================================================

output_path = "data/processed/test_risk_scores.csv"

risk_data.to_csv(
    output_path,
    index=False,
)


# ============================================================
# 7. Display summary
# ============================================================

print("\n" + "=" * 70)
print("PREDICTIVE MAINTENANCE RISK SCORING")
print("=" * 70)

print("\nRisk distribution:")
print(
    risk_data["risk_level"]
    .value_counts()
    .reindex(["Low", "Medium", "High"], fill_value=0)
)

print("\nRisk percentages:")
print(
    risk_data["risk_level"]
    .value_counts(normalize=True)
    .reindex(["Low", "Medium", "High"], fill_value=0)
    .mul(100)
    .round(2)
)

print("\nExample predictions:")
print(
    risk_data[["fault_probability", "risk_percentage", "risk_level"]]
    .head(10)
    .to_string(index=False)
)

print(f"\nSaved risk scores to: {output_path}")
