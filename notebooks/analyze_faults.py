import sqlite3

DB_PATH = "../data/raw/td_V2.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Number of fault-inducing commits
cursor.execute("""
    SELECT COUNT(DISTINCT FAULT_INDUCING_COMMIT_HASH)
    FROM SZZ_FAULT_INDUCING_COMMITS
""")

print("Unique fault-inducing commits:", cursor.fetchone()[0])


# Number of fault-fixing commits
cursor.execute("""
    SELECT COUNT(DISTINCT FAULT_FIXING_COMMIT_HASH)
    FROM SZZ_FAULT_INDUCING_COMMITS
""")

print("Unique fault-fixing commits:", cursor.fetchone()[0])


# How many fault-inducing commits exist in Git commits?
cursor.execute("""
    SELECT COUNT(DISTINCT s.FAULT_INDUCING_COMMIT_HASH)
    FROM SZZ_FAULT_INDUCING_COMMITS s
    INNER JOIN GIT_COMMITS g
        ON s.PROJECT_ID = g.PROJECT_ID
        AND s.FAULT_INDUCING_COMMIT_HASH = g.COMMIT_HASH
""")

print("Fault-inducing commits found in GIT_COMMITS:", cursor.fetchone()[0])


# How many of those have file-change information?
cursor.execute("""
    SELECT COUNT(DISTINCT s.FAULT_INDUCING_COMMIT_HASH)
    FROM SZZ_FAULT_INDUCING_COMMITS s
    INNER JOIN GIT_COMMITS_CHANGES c
        ON s.PROJECT_ID = c.PROJECT_ID
        AND s.FAULT_INDUCING_COMMIT_HASH = c.COMMIT_HASH
""")

print("Fault-inducing commits with file changes:", cursor.fetchone()[0])


conn.close()
