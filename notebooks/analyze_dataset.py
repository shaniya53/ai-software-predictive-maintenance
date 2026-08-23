import sqlite3

DB_PATH = "../data/raw/td_V2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("\n" + "=" * 70)
print("1. FAULT LABEL DISTRIBUTION")
print("=" * 70)

# Total commits
cursor.execute("""
    SELECT COUNT(DISTINCT PROJECT_ID || '|' || COMMIT_HASH)
    FROM GIT_COMMITS
""")
total_commits = cursor.fetchone()[0]

# Fault-inducing commits
cursor.execute("""
    SELECT COUNT(DISTINCT PROJECT_ID || '|' || FAULT_INDUCING_COMMIT_HASH)
    FROM SZZ_FAULT_INDUCING_COMMITS
""")
fault_commits = cursor.fetchone()[0]

non_fault_commits = total_commits - fault_commits

print("Total commits:", total_commits)
print("Fault-inducing commits:", fault_commits)
print("Non-fault-inducing commits:", non_fault_commits)

if total_commits:
    print(f"Fault-inducing percentage: " f"{fault_commits / total_commits * 100:.2f}%")


print("\n" + "=" * 70)
print("2. COMMIT DATE RANGE")
print("=" * 70)

cursor.execute("""
    SELECT MIN(COMMITTER_DATE), MAX(COMMITTER_DATE)
    FROM GIT_COMMITS
""")

date_range = cursor.fetchone()

print("Earliest commit:", date_range[0])
print("Latest commit:", date_range[1])


print("\n" + "=" * 70)
print("3. FAULT-INDUCING COMMIT DATE RANGE")
print("=" * 70)

cursor.execute("""
    SELECT MIN(g.COMMITTER_DATE), MAX(g.COMMITTER_DATE)
    FROM GIT_COMMITS g
    INNER JOIN SZZ_FAULT_INDUCING_COMMITS s
        ON g.PROJECT_ID = s.PROJECT_ID
        AND g.COMMIT_HASH = s.FAULT_INDUCING_COMMIT_HASH
""")

fault_dates = cursor.fetchone()

print("Earliest fault-inducing commit:", fault_dates[0])
print("Latest fault-inducing commit:", fault_dates[1])


print("\n" + "=" * 70)
print("4. FAULT COMMITS WITH REFACTORING")
print("=" * 70)

cursor.execute("""
    SELECT COUNT(DISTINCT s.PROJECT_ID || '|' || s.FAULT_INDUCING_COMMIT_HASH)
    FROM SZZ_FAULT_INDUCING_COMMITS s
    INNER JOIN REFACTORING_MINER r
        ON s.PROJECT_ID = r.PROJECT_ID
        AND s.FAULT_INDUCING_COMMIT_HASH = r.COMMIT_HASH
""")

fault_with_refactoring = cursor.fetchone()[0]

print("Fault-inducing commits with refactoring:", fault_with_refactoring)


print("\n" + "=" * 70)
print("5. FAULT COMMITS WITH SONAR")
print("=" * 70)

cursor.execute("""
    SELECT COUNT(DISTINCT s.PROJECT_ID || '|' || s.FAULT_INDUCING_COMMIT_HASH)
    FROM SZZ_FAULT_INDUCING_COMMITS s
    INNER JOIN SONAR_ANALYSIS sa
        ON s.PROJECT_ID = sa.PROJECT_ID
        AND s.FAULT_INDUCING_COMMIT_HASH = sa.REVISION
""")

fault_with_sonar = cursor.fetchone()[0]

print("Fault-inducing commits with Sonar:", fault_with_sonar)


print("\n" + "=" * 70)
print("6. FAULT COMMITS WITH JIRA")
print("=" * 70)

cursor.execute("""
    SELECT COUNT(DISTINCT s.PROJECT_ID || '|' || s.FAULT_INDUCING_COMMIT_HASH)
    FROM SZZ_FAULT_INDUCING_COMMITS s
    INNER JOIN JIRA_ISSUES j
        ON s.PROJECT_ID = j.PROJECT_ID
        AND s.FAULT_INDUCING_COMMIT_HASH = j.HASH
""")

fault_with_jira = cursor.fetchone()[0]

print("Fault-inducing commits with Jira:", fault_with_jira)


conn.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
