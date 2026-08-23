import sqlite3

DB_PATH = "../data/raw/td_V2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

query = """
SELECT
    sa.PROJECT_ID,
    sa.ANALYSIS_KEY,
    sa.REVISION,
    gc.COMMIT_HASH,
    gc.COMMIT_MESSAGE,
    gc.COMMITTER_DATE
FROM SONAR_ANALYSIS sa
INNER JOIN GIT_COMMITS gc
    ON sa.PROJECT_ID = gc.PROJECT_ID
    AND sa.REVISION = gc.COMMIT_HASH
LIMIT 10;
"""

cursor.execute(query)

rows = cursor.fetchall()

print("\n=== SONAR → GIT MATCHES ===")

for row in rows:
    print(row)

print("\nNumber of matches:", len(rows))

conn.close()
