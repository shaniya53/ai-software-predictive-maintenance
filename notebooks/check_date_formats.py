import sqlite3

DB_PATH = "../data/raw/td_V2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT COMMITTER_DATE
    FROM GIT_COMMITS
    LIMIT 20
""")

print("Sample raw commit dates:\n")

for row in cursor.fetchall():
    print(repr(row[0]))

conn.close()
