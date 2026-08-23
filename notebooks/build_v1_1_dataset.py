import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = "../data/raw/td_V2.db"
OUTPUT_PATH = "../data/processed/research_dataset_v1_1.csv"


print("Connecting to database...")
conn = sqlite3.connect(DB_PATH)


# ============================================================
# 1. GIT COMMITS
# ============================================================

print("Loading Git commits...")

commits = pd.read_sql_query(
    """
    SELECT
        PROJECT_ID,
        COMMIT_HASH,
        COMMITTER_DATE AS commit_date,
        AUTHOR,
        MERGE,
        IN_MAIN_BRANCH
    FROM GIT_COMMITS
""",
    conn,
)

# Keep original date information for reliable chronological ordering
commits["commit_date"] = pd.to_datetime(
    commits["commit_date"], format="mixed", utc=True, errors="coerce"
)

print("Commits loaded:", len(commits))
print("Unparsed commit dates:", commits["commit_date"].isna().sum())


# ============================================================
# 2. CURRENT COMMIT CHANGE FEATURES
# ============================================================

print("Building change features...")

changes = pd.read_sql_query(
    """
    SELECT
        PROJECT_ID,
        COMMIT_HASH,
        COUNT(DISTINCT FILE) AS files_changed,
        COALESCE(SUM(LINES_ADDED), 0) AS lines_added,
        COALESCE(SUM(LINES_REMOVED), 0) AS lines_removed
    FROM GIT_COMMITS_CHANGES
    GROUP BY PROJECT_ID, COMMIT_HASH
""",
    conn,
)

changes["total_lines_changed"] = changes["lines_added"] + changes["lines_removed"]

changes["avg_lines_per_file"] = changes["total_lines_changed"] / changes[
    "files_changed"
].replace(0, 1)


df = commits.merge(changes, on=["PROJECT_ID", "COMMIT_HASH"], how="left")

for col in [
    "files_changed",
    "lines_added",
    "lines_removed",
    "total_lines_changed",
    "avg_lines_per_file",
]:
    df[col] = df[col].fillna(0)


# ============================================================
# 3. SORT CHRONOLOGICALLY
# ============================================================

print("Sorting commits chronologically...")

df = df.sort_values(["PROJECT_ID", "commit_date", "COMMIT_HASH"]).reset_index(drop=True)


# ============================================================
# 4. PREVIOUS COMMIT COUNT
# ============================================================

print("Calculating previous project commits...")

df["previous_commit_count"] = df.groupby("PROJECT_ID").cumcount()


# ============================================================
# 5. PREVIOUS AUTHOR COMMIT COUNT
# ============================================================

print("Calculating previous author commits...")

df["previous_author_commit_count"] = df.groupby(["PROJECT_ID", "AUTHOR"]).cumcount()


# ============================================================
# 6. FILE HISTORY
# ============================================================

print("Loading file history...")

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

file_changes["DATE"] = pd.to_datetime(
    file_changes["DATE"], format="mixed", utc=True, errors="coerce"
)

file_changes = file_changes.sort_values(["PROJECT_ID", "FILE", "DATE", "COMMIT_HASH"])

# Number of earlier changes to the same file
file_changes["previous_file_changes"] = file_changes.groupby(
    ["PROJECT_ID", "FILE"]
).cumcount()

# Date of previous change to same file
file_changes["previous_file_change_date"] = file_changes.groupby(
    ["PROJECT_ID", "FILE"]
)["DATE"].shift(1)

file_changes["days_since_previous_change"] = (
    file_changes["DATE"] - file_changes["previous_file_change_date"]
).dt.total_seconds() / 86400

# First recorded file change
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
# 7. REFACTORING FEATURES
# ============================================================

print("Building refactoring features...")

refactoring = pd.read_sql_query(
    """
    SELECT
        PROJECT_ID,
        COMMIT_HASH,
        COUNT(*) AS refactoring_count,
        COUNT(DISTINCT REFACTORING_TYPE)
            AS refactoring_type_count
    FROM REFACTORING_MINER
    GROUP BY PROJECT_ID, COMMIT_HASH
""",
    conn,
)

refactoring["has_refactoring"] = 1

df = df.merge(refactoring, on=["PROJECT_ID", "COMMIT_HASH"], how="left")

df["has_refactoring"] = df["has_refactoring"].fillna(0).astype(int)

df["refactoring_count"] = df["refactoring_count"].fillna(0)

df["refactoring_type_count"] = df["refactoring_type_count"].fillna(0)


# ============================================================
# 8. SZZ TARGET
# ============================================================

print("Loading SZZ fault information...")

faults = pd.read_sql_query(
    """
    SELECT DISTINCT
        PROJECT_ID,
        FAULT_INDUCING_COMMIT_HASH
    FROM SZZ_FAULT_INDUCING_COMMITS
""",
    conn,
)

faults = faults.rename(columns={"FAULT_INDUCING_COMMIT_HASH": "COMMIT_HASH"})

faults["fault_inducing"] = 1

df = df.merge(faults, on=["PROJECT_ID", "COMMIT_HASH"], how="left")

df["fault_inducing"] = df["fault_inducing"].fillna(0).astype(int)


# ============================================================
# 9. PREVIOUS FAULT COUNT
# ============================================================

print("Calculating previous fault count...")

# Create a set of fault-inducing commit keys
fault_keys = set(zip(faults["PROJECT_ID"], faults["COMMIT_HASH"]))

previous_fault_counts = []
fault_count_by_project = {}

for _, row in df.iterrows():

    project = row["PROJECT_ID"]
    commit = row["COMMIT_HASH"]

    # Number of faults observed BEFORE this commit
    previous_fault_counts.append(fault_count_by_project.get(project, 0))

    # Only AFTER assigning the current value do we update history
    if (project, commit) in fault_keys:
        fault_count_by_project[project] = fault_count_by_project.get(project, 0) + 1

df["previous_fault_count"] = previous_fault_counts


# ============================================================
# 10. FINAL COLUMN ORDER
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
    "previous_fault_count",
    "has_refactoring",
    "refactoring_count",
    "refactoring_type_count",
    "fault_inducing",
]

df = df[final_columns]


# ============================================================
# 11. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("V1.1 VALIDATION")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nTarget distribution:")
print(df["fault_inducing"].value_counts())

print("\nMissing values:")
print(df.isna().sum())

print("\nPrevious fault count:")
print(df["previous_fault_count"].describe())


# ============================================================
# 12. SAVE
# ============================================================

output_path = Path(OUTPUT_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("\nSaved to:")
print(output_path)

conn.close()

print("\nV1.1 DATASET COMPLETE!")
