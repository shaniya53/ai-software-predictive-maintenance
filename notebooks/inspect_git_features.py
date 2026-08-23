import sqlite3

DB_PATH = "../data/raw/td_V2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("\n=== GIT_COMMITS COLUMNS ===")

cursor.execute("""
    PRAGMA table_info("GIT_COMMITS")
""")

for column in cursor.fetchall():
    print(column)

print("\n=== GIT_COMMITS_CHANGES COLUMNS ===")

cursor.execute("""
    PRAGMA table_info("GIT_COMMITS_CHANGES")
""")

for column in cursor.fetchall():
    print(column)

conn.close()
