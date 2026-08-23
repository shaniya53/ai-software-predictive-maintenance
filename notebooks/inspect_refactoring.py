import sqlite3

DB_PATH = "../data/raw/td_V2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Show columns
cursor.execute("""
    PRAGMA table_info("REFACTORING_MINER")
""")

columns = cursor.fetchall()

print("\n=== REFACTORING_MINER COLUMNS ===")
for column in columns:
    print(column)

# Show sample records
cursor.execute("""
    SELECT *
    FROM REFACTORING_MINER
    LIMIT 10
""")

rows = cursor.fetchall()

print("\n=== SAMPLE REFACTORINGS ===")
for row in rows:
    print(row)

# Count total records
cursor.execute("""
    SELECT COUNT(*)
    FROM REFACTORING_MINER
""")

print("\nTotal refactoring records:", cursor.fetchone()[0])

conn.close()
