import sqlite3

DB_PATH = "../data/raw/td_V2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1. List all tables
cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    ORDER BY name;
""")

tables = cursor.fetchall()

print("\n=== TABLES ===")
for table in tables:
    print(table[0])

# 2. Show columns and row count for every table
for (table_name,) in tables:
    print(f"\n=== {table_name} ===")

    cursor.execute(f'PRAGMA table_info("{table_name}")')
    columns = cursor.fetchall()

    print("Columns:")
    for column in columns:
        print(f"  - {column[1]} ({column[2]})")

    cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    count = cursor.fetchone()[0]

    print(f"Rows: {count}")

conn.close()
