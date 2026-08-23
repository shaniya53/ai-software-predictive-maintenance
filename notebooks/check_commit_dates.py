import sqlite3

DB_PATH = "../data/raw/td_V2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT
        COUNT(*) AS total,
        SUM(
            CASE
                WHEN COMMITTER_DATE IS NULL
                     OR TRIM(COMMITTER_DATE) = ''
                THEN 1
                ELSE 0
            END
        ) AS missing_committer_date,
        SUM(
            CASE
                WHEN AUTHOR_DATE IS NULL
                     OR TRIM(AUTHOR_DATE) = ''
                THEN 1
                ELSE 0
            END
        ) AS missing_author_date
    FROM GIT_COMMITS
""")

total, missing_committer, missing_author = cursor.fetchone()

print("Total commits:", total)
print("Missing committer dates:", missing_committer)
print("Missing author dates:", missing_author)

cursor.execute("""
    SELECT
        COMMIT_HASH,
        COMMITTER_DATE,
        AUTHOR_DATE
    FROM GIT_COMMITS
    WHERE COMMITTER_DATE IS NULL
       OR TRIM(COMMITTER_DATE) = ''
    LIMIT 10
""")

print("\nSample commits with missing committer dates:")

for row in cursor.fetchall():
    print(row)

conn.close()
