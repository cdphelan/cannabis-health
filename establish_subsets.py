import sqlite3
import pandas as pd
import json
import random
from datetime import datetime

# === Paths ===
DB_PATH = "../../data_collection/reddit_combined.db"
GOLD_LABEL_PATH = "data/silver_standard_codes.csv"
ANNOTATED_JSONL = "annotated_comments.jsonl"

# === Parameters ===
NUM_RANDOM_KFAILS = 10000
NUM_KFAILS_FOR_GPT = 500
NUM_PASSED_FOR_GPT = 8460
DATE_NOW = datetime.now().strftime("%Y-%m-%d")

# === Connect to DB ===
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# === Load gold_label ===
gold_df = pd.read_csv(GOLD_LABEL_PATH)
gold_ids = set(gold_df["id"].astype(str))

# === Get cannabis-labeled post IDs ===
cur.execute("SELECT id FROM posts WHERE category = 'cannabis'")
post_ids = set(row[0] for row in cur.fetchall())

# === Get comment IDs where the parent post is cannabis-labeled ===
cur.execute("""
    SELECT c.id 
    FROM comments c
    JOIN posts p ON c.post_id = p.id
    WHERE p.category = 'cannabis'
""")
comment_ids = set(row[0] for row in cur.fetchall())

# === Combine all cannabis-related IDs ===
all_ids = post_ids.union(comment_ids)

# === Load cannabis-only keyword_matches IDs ===
cur.execute("""
    SELECT DISTINCT km.id
    FROM keyword_matches km
    LEFT JOIN posts p ON km.id = p.id
    LEFT JOIN comments c ON km.id = c.id
    LEFT JOIN posts parent_post ON c.post_id = parent_post.id
    WHERE 
        (p.category = 'cannabis' OR parent_post.category = 'cannabis')
""")
matched_ids = set(row[0] for row in cur.fetchall())


# === Random keyword-fails (not in keyword_matches) ===
non_matched_ids = list(all_ids - matched_ids)
random_kfails = set(random.sample(non_matched_ids, NUM_RANDOM_KFAILS))

# === Load annotated_comments.jsonl ===
annotated_ids = set()
with open(ANNOTATED_JSONL, "r") as f:
    for line in f:
        entry = json.loads(line)
        annotated_ids.add(entry["id"])

# === GPT random set ===
# Combine:
# 1. All gold_label
# 2. 500 from keyword-fails
# 3. annotated ids
# 4. ~8k random from keyword matches

gpt_ids = set(gold_ids)
gpt_ids.update(random.sample(list(random_kfails), NUM_KFAILS_FOR_GPT))
gpt_ids.update(annotated_ids)
gpt_ids.update(random.sample(list(matched_ids - gold_ids - annotated_ids), NUM_PASSED_FOR_GPT))

# === Create membership table ===
records = []

for _id in gold_ids:
    records.append((_id, "gold_label", DATE_NOW))

for _id in random_kfails:
    records.append((_id, "random_kfails", DATE_NOW))

for _id in gpt_ids:
    records.append((_id, "gpt_random_set", DATE_NOW))

# NOTE: `full_text_sample` not included yet — to be added later

print("Gold:", len(gold_ids))
print("Random k-fails:", len(random_kfails))
print("Annotated:", len(annotated_ids))
print("Keyword-matches (eligible for GPT):", len(matched_ids - gold_ids - annotated_ids))
print("GPT total (after combining):", len(gpt_ids))

# === Save to database ===
cur.execute("DROP TABLE IF EXISTS subset_membership;")
conn.commit()

cur.execute("""
CREATE TABLE IF NOT EXISTS subset_membership (
    id TEXT,
    subset TEXT,
    date_created TEXT,
    PRIMARY KEY (id, subset)
)
""")

cur.executemany("""
INSERT OR IGNORE INTO subset_membership (id, subset, date_created)
VALUES (?, ?, ?)
""", records)

conn.commit()
conn.close()

print(f"✅ Inserted {len(records)} records into subset_membership.")
