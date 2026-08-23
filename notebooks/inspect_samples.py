import sqlite3

DB_PATH = "../data/raw/td_V2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

tables = [
    "PROJECTS",
    "GIT_COMMITS",
    "GIT_COMMITS_CHANGES",
    "SONAR_ANALYSIS",
    "SONAR_MEASURES",
    "SZZ_FAULT_INDUCING_COMMITS",
]

for table in tables:
    print("\n" + "=" * 80)
    print(f"TABLE: {table}")
    print("=" * 80)

    cursor.execute(f'SELECT * FROM "{table}" LIMIT 3')

    rows = cursor.fetchall()

    # Get column names
    column_names = [description[0] for description in cursor.description]

    print("\nColumns:")
    print(column_names)

    print("\nSample rows:")
    for row in rows:
        print(row)

conn.close()
