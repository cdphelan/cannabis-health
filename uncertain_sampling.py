import pandas as pd
import numpy as np
import sqlite3

# SAMPLE_SIZE = 5 #minimum for each keyword and subreddit

# # === Load data ===
# df_original = pd.read_csv("data/allpreds_csv_data.csv", low_memory=False)
# df = df_original.copy()

# # === Filter out gold label and xgboost entries ===
# df = df[(df["subset"] != "gold_label") & (df["model"] != "xgboost")].copy()

# # === Rescale GPT model confidence to 0.5–1 ===
# df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
# gpt_mask = df["model"].isin(["gpt_zero_shot", "gpt_few_shot"])
# df.loc[gpt_mask, "confidence"] = 0.5 + 0.5 * df.loc[gpt_mask, "confidence"]

# # === Group by (id, subreddit, keyword) and calculate disagreement/uncertainty ===
# # Explode comma-separated keywords
# df["keyword"] = df["keyword"].astype(str).str.lower().str.split(",")
# df = df.explode("keyword")
# df["keyword"] = df["keyword"].str.strip()

# # Now safe to group
# grouped = df.groupby(["id", "subreddit", "keyword"])

# def assess_group(gr):
#     relevance_counts = gr["relevance"].value_counts(normalize=True).to_dict()
#     p = relevance_counts.get(1, 0)
#     disagreement = int((p >= 0.3 and p <= 0.7))

#     avg_conf = gr["confidence"].mean()
#     if pd.isna(avg_conf):
#         uncertainty = 0.0  # Treat NaNs as no uncertainty contribution
#     elif 0.5 <= avg_conf < 0.65:
#         uncertainty = 0.5
#     elif 0.65 <= avg_conf < 0.75:
#         uncertainty = 1.0
#     else:
#         uncertainty = 0.0

#     priority_score = disagreement + uncertainty
#     return pd.Series({
#         "priority_score": priority_score,
#         "disagreement": disagreement,
#         "uncertainty": uncertainty,
#         "avg_conf": avg_conf,
#         "relevance_split": p
#     })

# scored = grouped.apply(assess_group).reset_index()

# # === Stratified sampling ===
# # Ensure each keyword and subreddit has at least SAMPLE_SIZE examples
# scored["keyword"] = scored["keyword"].str.strip().str.lower()
# scored["subreddit"] = scored["subreddit"].str.strip().str.lower()

# # Explode for stratified sampling
# kw_sample = (
#     scored.groupby("keyword", group_keys=False)
#     .apply(lambda g: g.sort_values("priority_score", ascending=False).head(SAMPLE_SIZE))
# )

# sub_sample = (
#     scored.groupby("subreddit", group_keys=False)
#     .apply(lambda g: g.sort_values("priority_score", ascending=False).head(SAMPLE_SIZE))
# )

# # === Combine and deduplicate ===
# final_sample_ids = pd.concat([kw_sample, sub_sample], ignore_index=True).drop_duplicates(subset=["id"])

# # Merge with original df to restore original keywords
# original_keywords = df_original[["id", "keyword"]].drop_duplicates(subset=["id"])
# final_sample = final_sample_ids.drop(columns=["keyword"]).merge(original_keywords, on="id", how="left")


# # === Output sample ===
# final_sample.to_csv("data/high_value_stratified_sample.csv", index=False)
# print("Sample saved to data/high_value_stratified_sample.csv")


# Load your stratified sample
# sample_df = final_sample
sample_df = pd.read_csv("data/high_value_stratified_sample.csv")


# Connect to your SQLite database
conn = sqlite3.connect("../../data_collection/reddit_combined.db")  

# Get unique IDs from sample
sample_ids = tuple(sample_df["id"].unique())

# Query post texts
query_posts = f"""
    SELECT id, title, selftext
    FROM posts
    WHERE id IN ({','.join(['?'] * len(sample_ids))})
"""
df_posts = pd.read_sql_query(query_posts, conn, params=sample_ids)
df_posts["text"] = df_posts["title"].fillna('') + " | " + df_posts["selftext"].fillna('')
df_posts = df_posts[["id", "text"]]

# Query comment texts
query_comments = f"""
    SELECT id, body AS text
    FROM comments
    WHERE id IN ({','.join(['?'] * len(sample_ids))})
"""
df_comments = pd.read_sql_query(query_comments, conn, params=sample_ids)

# Combine posts and comments
df_text = pd.concat([df_posts, df_comments], ignore_index=True)

# Merge with sample
merged = sample_df.merge(df_text, on="id", how="left")

# Output
print(merged[["id", "subreddit", "keyword", "priority_score", "text"]].head(3))

# Optionally save to CSV
merged.to_csv("data/high_value_stratified_with_text.csv", index=False)
