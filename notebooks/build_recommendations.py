import pandas as pd

# ============================================================
# 1. Load risk-scored data
# ============================================================

data = pd.read_csv("data/processed/test_risk_scores.csv")


# ============================================================
# 2. Generate maintenance recommendations
# ============================================================


def generate_recommendations(row):
    recommendations = []

    # Large code change
    if row["total_lines_changed"] > 157:
        recommendations.append("Review the large code change carefully")

    # Many files changed
    if row["files_changed"] > 3:
        recommendations.append("Review interactions between the changed files")

    # High previous fault history
    if row["previous_fault_count"] > 319:
        recommendations.append("Inspect historically fault-prone areas")

    # Refactoring
    if row["has_refactoring"] == 1:
        recommendations.append("Perform additional testing after refactoring")

    # Frequent project activity
    if row["previous_commit_count"] > 1956:
        recommendations.append("Review the change against recent project history")

    # No specific recommendation
    if not recommendations:
        if row["risk_level"] == "High":
            recommendations.append("Perform detailed code review and testing")
        elif row["risk_level"] == "Medium":
            recommendations.append("Consider additional code review and testing")
        else:
            recommendations.append("Continue with standard review and testing")

    return " | ".join(recommendations)


data["maintenance_recommendation"] = data.apply(generate_recommendations, axis=1)


# ============================================================
# 3. Save recommendations
# ============================================================

output_path = "data/processed/test_risk_recommendations.csv"

data.to_csv(output_path, index=False)


# ============================================================
# 4. Summary
# ============================================================

print("\n" + "=" * 70)
print("MAINTENANCE RECOMMENDATION SYSTEM")
print("=" * 70)

print("\nRecommendation counts:")

recommendation_counts = data["maintenance_recommendation"].value_counts()

print(recommendation_counts.head(10).to_string())


print("\nExample recommendations:")

print(
    data[["fault_probability", "risk_level", "maintenance_recommendation"]]
    .head(10)
    .to_string(index=False)
)

print(f"\nSaved recommendations to: {output_path}")
