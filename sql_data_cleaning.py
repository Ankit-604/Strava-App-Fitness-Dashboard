"""
SQL-based data cleaning & integrity checks
===========================================
Mirrors the checks a reviewer expects (per the case-study brief): using SQL (here, Python's
built-in sqlite3 - same syntax as DB Browser for SQLite) to check for blanks, negative values,
and impossible values BEFORE analysis. Run this after merge_data.py.
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect(':memory:')
daily = pd.read_csv('daily_merged.csv', parse_dates=['Date'])
daily.to_sql('daily', conn, index=False)

def run(title, sql):
    print(f'\n--- {title} ---')
    print(sql.strip())
    result = pd.read_sql_query(sql, conn)
    print(f'-> {len(result)} row(s) flagged')
    if len(result):
        print(result.head(10))
    return result

# 1) Check for blank/NULL values in core activity columns
run("Check for NULLs in core columns", """
    SELECT * FROM daily
    WHERE TotalSteps IS NULL OR Calories IS NULL OR SedentaryMinutes IS NULL
""")

# 2) Check for negative values (impossible for steps, distance, calories, minutes)
run("Check for negative values", """
    SELECT Id, Date, TotalSteps, TotalDistance, Calories
    FROM daily
    WHERE TotalSteps < 0 OR TotalDistance < 0 OR Calories < 0
""")

# 3) Check for sedentary minutes >= 1440 (a full day has only 1440 minutes total -
#    if Sedentary alone hits that, the device logged with no real activity all day,
#    which is a device-not-worn day, not a data error, but worth flagging)
run("Days flagged: Sedentary minutes >= 1440 (full day, tracker likely not worn)", """
    SELECT Id, Date, SedentaryMinutes
    FROM daily
    WHERE SedentaryMinutes >= 1440
""")

# 4) Check that the four intensity-minute columns never sum to more than 1440/day
run("Days where total logged minutes exceed 1440 (data error if so)", """
    SELECT Id, Date,
           (SedentaryMinutes + LightlyActiveMinutes + FairlyActiveMinutes + VeryActiveMinutes) AS total_minutes
    FROM daily
    WHERE (SedentaryMinutes + LightlyActiveMinutes + FairlyActiveMinutes + VeryActiveMinutes) > 1440
""")

# 5) Check for duplicate Id+Date rows (each user should have at most 1 row per day)
run("Duplicate Id+Date rows", """
    SELECT Id, Date, COUNT(*) as n
    FROM daily
    GROUP BY Id, Date
    HAVING COUNT(*) > 1
""")

# 6) Sanity-check the user count and date range match what the dataset documentation claims
n_users = pd.read_sql_query("SELECT COUNT(DISTINCT Id) as n FROM daily", conn).iloc[0]['n']
date_range = pd.read_sql_query("SELECT MIN(Date) as start, MAX(Date) as end FROM daily", conn).iloc[0]
print(f"\n--- Coverage check ---\nUsers: {n_users}\nDate range: {date_range['start']} to {date_range['end']}")

conn.close()
print("\nCleaning checks complete. See README/report for how each finding was handled.")
