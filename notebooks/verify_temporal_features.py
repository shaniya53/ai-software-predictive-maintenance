import sqlite3
import pandas as pd

DB_PATH = "../data/raw/td_V2.db"
V1_PATH = "../data/processed/research_dataset_v1_1.csv"

print("=" * 70)
print("INDEPENDENT TEMPORAL FEATURE VERIFICATION")
print("=" * 70)


# ============================================================
# 1. LOAD V1.1
# ============================================================

df = pd.read_csv(V1_PATH)

df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True, errors="coerce")

print("V1.1 rows:", len(df))


# ============================================================
# 2. LOAD RAW GIT COMMITS
# ============================================================

print("\nLoading raw Git commits...")

conn = sqlite3.connect(DB_PATH)

commits = pd.read_sql_query(
    """
    SELECT
        PROJECT_ID,
        COMMIT_HASH,
        COMMITTER_DATE AS commit_date,
        AUTHOR
    FROM GIT_COMMITS
    """,
    conn,
)

commits["commit_date"] = pd.to_datetime(
    commits["commit_date"], format="mixed", utc=True, errors="coerce"
)

commits = commits.sort_values(["PROJECT_ID", "commit_date", "COMMIT_HASH"]).reset_index(
    drop=True
)


# ============================================================
# 3. INDEPENDENT PREVIOUS COMMIT COUNT
# ============================================================

print("Calculating independent previous commit count...")

commits["expected_previous_commit_count"] = commits.groupby("PROJECT_ID").cumcount()


# ============================================================
# 4. INDEPENDENT PREVIOUS AUTHOR COUNT
# ============================================================

print("Calculating independent previous author count...")

commits["expected_previous_author_commit_count"] = commits.groupby(
    ["PROJECT_ID", "AUTHOR"]
).cumcount()


# ============================================================
# 5. COMPARE WITH V1.1
# ============================================================

print("\n" + "=" * 70)
print("COMPARING HISTORICAL FEATURES")
print("=" * 70)

comparison = df.merge(
    commits[
        [
            "PROJECT_ID",
            "COMMIT_HASH",
            "expected_previous_commit_count",
            "expected_previous_author_commit_count",
        ]
    ],
    on=["PROJECT_ID", "COMMIT_HASH"],
    how="left",
)

commit_count_mismatch = (
    comparison["previous_commit_count"] != comparison["expected_previous_commit_count"]
)

author_count_mismatch = (
    comparison["previous_author_commit_count"]
    != comparison["expected_previous_author_commit_count"]
)

print("Previous commit count mismatches:", commit_count_mismatch.sum())

print("Previous author commit count mismatches:", author_count_mismatch.sum())


# ============================================================
# 6. VERIFY PREVIOUS FAULT COUNT INDEPENDENTLY
# ============================================================

print("\nLoading SZZ fault labels...")

faults = pd.read_sql_query(
    """
    SELECT DISTINCT
        PROJECT_ID,
        FAULT_INDUCING_COMMIT_HASH
    FROM SZZ_FAULT_INDUCING_COMMITS
    """,
    conn,
)

fault_keys = set(zip(faults["PROJECT_ID"], faults["FAULT_INDUCING_COMMIT_HASH"]))

print("Fault-inducing commits:", len(fault_keys))


# Calculate independently from chronological commits

fault_count_by_project = {}
expected_previous_fault_count = []

for _, row in commits.iterrows():

    project = row["PROJECT_ID"]
    commit = row["COMMIT_HASH"]

    # IMPORTANT:
    # Read history BEFORE adding current commit
    expected_previous_fault_count.append(fault_count_by_project.get(project, 0))

    if (project, commit) in fault_keys:
        fault_count_by_project[project] = fault_count_by_project.get(project, 0) + 1

commits["expected_previous_fault_count"] = expected_previous_fault_count


# ============================================================
# 7. COMPARE PREVIOUS FAULT COUNT
# ============================================================

comparison = df.merge(
    commits[["PROJECT_ID", "COMMIT_HASH", "expected_previous_fault_count"]],
    on=["PROJECT_ID", "COMMIT_HASH"],
    how="left",
)

fault_count_mismatch = (
    comparison["previous_fault_count"] != comparison["expected_previous_fault_count"]
)

print("Previous fault count mismatches:", fault_count_mismatch.sum())


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

total_mismatches = (
    commit_count_mismatch.sum()
    + author_count_mismatch.sum()
    + fault_count_mismatch.sum()
)

print("Total historical-feature mismatches:", total_mismatches)

if total_mismatches == 0:
    print("\nRESULT: ALL HISTORICAL FEATURES MATCH.")
    print("Temporal feature construction is independently verified.")
else:
    print("\nRESULT: MISMATCHES FOUND.")
    print("We need to investigate before proceeding.")


conn.close()

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
