import sqlite3
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "../data/raw/td_V2.db"
OUTPUT_PATH = "../data/processed/research_dataset_v1.csv"


# ============================================================
# CONNECT TO DATABASE
# ============================================================

print("Connecting to database...")

conn = sqlite3.connect(DB_PATH)


# ============================================================
# 1. AGGREGATE CURRENT COMMIT CHANGES
# ============================================================

print("Extracting commit change information...")

changes_query = """
SELECT
    PROJECT_ID,
    COMMIT_HASH,
    COUNT(DISTINCT FILE) AS files_changed,
    COALESCE(SUM(LINES_ADDED), 0) AS lines_added,
    COALESCE(SUM(LINES_REMOVED), 0) AS lines_removed
FROM GIT_COMMITS_CHANGES
GROUP BY PROJECT_ID, COMMIT_HASH
"""

changes = pd.read_sql_query(changes_query, conn)

changes["total_lines_changed"] = changes["lines_added"] + changes["lines_removed"]

changes["avg_lines_per_file"] = changes["total_lines_changed"] / changes[
    "files_changed"
].replace(0, 1)


# ============================================================
# 2. GET BASIC COMMIT INFORMATION
# ============================================================

print("Extracting commit information...")

commits_query = """
SELECT
    PROJECT_ID,
    COMMIT_HASH,
    COMMITTER_DATE AS commit_date,
    AUTHOR,
    MERGE,
    IN_MAIN_BRANCH
FROM GIT_COMMITS
"""

commits = pd.read_sql_query(commits_query, conn)


# ============================================================
# 3. COMBINE COMMITS + CHANGES
# ============================================================

print("Combining commit and change information...")

df = commits.merge(changes, on=["PROJECT_ID", "COMMIT_HASH"], how="left")

# Commits with no recorded file changes
df["files_changed"] = df["files_changed"].fillna(0)
df["lines_added"] = df["lines_added"].fillna(0)
df["lines_removed"] = df["lines_removed"].fillna(0)
df["total_lines_changed"] = df["total_lines_changed"].fillna(0)
df["avg_lines_per_file"] = df["avg_lines_per_file"].fillna(0)


# ============================================================
# 4. SORT CHRONOLOGICALLY
# ============================================================

print("Sorting commits chronologically...")

df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True, errors="coerce")

df = df.sort_values(["PROJECT_ID", "commit_date"]).reset_index(drop=True)


# ============================================================
# 5. HISTORICAL COMMIT COUNT
# ============================================================

print("Calculating historical commit counts...")

df["previous_commit_count"] = df.groupby("PROJECT_ID").cumcount()


# ============================================================
# 6. HISTORICAL AUTHOR COMMIT COUNT
# ============================================================

print("Calculating historical author commit counts...")

df["previous_author_commit_count"] = df.groupby(["PROJECT_ID", "AUTHOR"]).cumcount()


# ============================================================
# 7. HISTORICAL FILE CHANGES
# ============================================================

print("Calculating historical file-change information...")

# Build file-level history
file_changes = pd.read_sql_query(
    """
    SELECT
        PROJECT_ID,
        FILE,
        COMMIT_HASH,
        DATE
    FROM GIT_COMMITS_CHANGES
    """,
    conn,
)

file_changes["DATE"] = pd.to_datetime(file_changes["DATE"], utc=True, errors="coerce")

file_changes = file_changes.sort_values(["PROJECT_ID", "FILE", "DATE"])

# Number of previous changes for each file
file_changes["previous_file_changes"] = file_changes.groupby(
    ["PROJECT_ID", "FILE"]
).cumcount()

# Previous change date for each file
file_changes["previous_file_change_date"] = file_changes.groupby(
    ["PROJECT_ID", "FILE"]
)["DATE"].shift(1)

file_changes["days_since_previous_change"] = (
    file_changes["DATE"] - file_changes["previous_file_change_date"]
).dt.total_seconds() / 86400

# Missing previous date means this is the first recorded change
file_changes["days_since_previous_change"] = file_changes[
    "days_since_previous_change"
].fillna(-1)

# Aggregate file history to commit level
file_history = (
    file_changes.groupby(["PROJECT_ID", "COMMIT_HASH"])
    .agg(
        previous_file_changes=("previous_file_changes", "sum"),
        days_since_previous_change=("days_since_previous_change", "mean"),
    )
    .reset_index()
)

df = df.merge(file_history, on=["PROJECT_ID", "COMMIT_HASH"], how="left")

df["previous_file_changes"] = df["previous_file_changes"].fillna(0)

df["days_since_previous_change"] = df["days_since_previous_change"].fillna(-1)


# ============================================================
# 8. REFACTORING FEATURES
# ============================================================

print("Extracting refactoring features...")

refactoring_query = """
SELECT
    PROJECT_ID,
    COMMIT_HASH,
    COUNT(*) AS refactoring_count,
    COUNT(DISTINCT REFACTORING_TYPE)
        AS refactoring_type_count
FROM REFACTORING_MINER
GROUP BY PROJECT_ID, COMMIT_HASH
"""

refactoring = pd.read_sql_query(refactoring_query, conn)

refactoring["has_refactoring"] = 1

df = df.merge(refactoring, on=["PROJECT_ID", "COMMIT_HASH"], how="left")

df["has_refactoring"] = df["has_refactoring"].fillna(0).astype(int)

df["refactoring_count"] = df["refactoring_count"].fillna(0)

df["refactoring_type_count"] = df["refactoring_type_count"].fillna(0)


# ============================================================
# 9. SZZ TARGET
# ============================================================

print("Creating fault-inducing target...")

faults_query = """
SELECT DISTINCT
    PROJECT_ID,
    FAULT_INDUCING_COMMIT_HASH
FROM SZZ_FAULT_INDUCING_COMMITS
"""

faults = pd.read_sql_query(faults_query, conn)

faults["fault_inducing"] = 1

faults = faults.rename(columns={"FAULT_INDUCING_COMMIT_HASH": "COMMIT_HASH"})

df = df.merge(
    faults[["PROJECT_ID", "COMMIT_HASH", "fault_inducing"]],
    on=["PROJECT_ID", "COMMIT_HASH"],
    how="left",
)

df["fault_inducing"] = df["fault_inducing"].fillna(0).astype(int)


# ============================================================
# 10. SELECT FINAL V1 FEATURES
# ============================================================

final_columns = [
    "PROJECT_ID",
    "COMMIT_HASH",
    "commit_date",
    "lines_added",
    "lines_removed",
    "files_changed",
    "total_lines_changed",
    "avg_lines_per_file",
    "MERGE",
    "IN_MAIN_BRANCH",
    "previous_commit_count",
    "previous_author_commit_count",
    "previous_file_changes",
    "days_since_previous_change",
    "has_refactoring",
    "refactoring_count",
    "refactoring_type_count",
    "fault_inducing",
]

df = df[final_columns]


# ============================================================
# 11. SAVE DATASET
# ============================================================

output_path = Path(OUTPUT_PATH)

output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V1 DATASET CREATED")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nTarget distribution:")
print(df["fault_inducing"].value_counts())

print("\nTarget percentage:")
print(df["fault_inducing"].value_counts(normalize=True).mul(100).round(2))

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved to:")
print(output_path)

conn.close()

print("\nDone!")
