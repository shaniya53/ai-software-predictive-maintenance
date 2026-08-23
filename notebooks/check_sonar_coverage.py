import sqlite3

DB_PATH = "../data/raw/td_V2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Total Git commits
cursor.execute("""
    SELECT COUNT(DISTINCT PROJECT_ID || '|' || COMMIT_HASH)
    FROM GIT_COMMITS
""")
total_commits = cursor.fetchone()[0]

# Git commits that have a matching Sonar analysis
cursor.execute("""
    SELECT COUNT(DISTINCT gc.PROJECT_ID || '|' || gc.COMMIT_HASH)
    FROM GIT_COMMITS gc
    INNER JOIN SONAR_ANALYSIS sa
        ON gc.PROJECT_ID = sa.PROJECT_ID
        AND gc.COMMIT_HASH = sa.REVISION
""")
matched_commits = cursor.fetchone()[0]

print("Total Git commits:", total_commits)
print("Git commits with Sonar analysis:", matched_commits)

if total_commits > 0:
    percentage = matched_commits / total_commits * 100
    print(f"Coverage: {percentage:.2f}%")

# Number of Sonar analyses
cursor.execute("""
    SELECT COUNT(*)
    FROM SONAR_ANALYSIS
""")
sonar_analyses = cursor.fetchone()[0]

print("Total Sonar analyses:", sonar_analyses)

conn.close()
