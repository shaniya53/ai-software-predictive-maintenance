import sqlite3

DB_PATH = "../data/raw/td_V2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    PRAGMA table_info("JIRA_ISSUES")
""")

print("\n=== JIRA_ISSUES COLUMNS ===")

for column in cursor.fetchall():
    print(column)

cursor.execute("""
    SELECT *
    FROM JIRA_ISSUES
    LIMIT 10
""")

print("\n=== SAMPLE JIRA ISSUES ===")

for row in cursor.fetchall():
    print(row)

cursor.execute("""
    SELECT COUNT(*)
    FROM JIRA_ISSUES
""")

print("\nTotal Jira issues:", cursor.fetchone()[0])

conn.close()
