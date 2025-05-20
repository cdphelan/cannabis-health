import pandas as pd
import os
import re

# Load a dictionary mapping regex patterns to canonical keyword labels
from keywords_pain_sleep_anxiety import pain_sleep_anxiety as regex_to_keyword

# Path to directory containing Excel files from different subreddits
input_dir = "../Raw"

# Extract regex patterns from the keyword dictionary
regex_patterns = list(regex_to_keyword.keys())

# The text fields to scan for keyword matches
text_columns = ["title", "post_text", "comment_body", "reply_body"]

# ------------------------------
# OPTION 1: STRUCTURE-PRESERVING FILTERING 
# ------------------------------
def filter_by_row():
    """
    Filters input rows by applying regex keyword matching to each text field.
    Tracks previously matched content to avoid duplicate hits across structural labels
    (e.g., a reply showing up again as a comment).
    Returns a DataFrame of matched rows with an additional 'keyword' column.
    """
    matched_fields = {col: set() for col in text_columns}
    matched_rows = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(input_dir, filename)
            try:
                df = pd.read_excel(filepath)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            if not all(col in df.columns for col in text_columns):
                print(f"Skipping {filename} due to missing columns.")
                continue

            df['subreddit'] = filename.replace(".xlsx", "")

            for _, row in df.iterrows():
                row_matches = []
                post_id = row.get("post_id")
                comment_id = row.get("comment_id")
                reply_id = row.get("reply_id")

                for col in text_columns:
                    id_key = None
                    if col in ["title", "post_text"]:
                        id_key = post_id
                    elif col == "comment_body":
                        id_key = comment_id
                    elif col == "reply_body":
                        id_key = reply_id

                    if pd.isna(row[col]) or (id_key, col) in matched_fields[col]:
                        continue

                    text = str(row[col])
                    matches = [regex_to_keyword[regex] for regex in regex_patterns if re.search(regex, text, re.IGNORECASE)]
                    if matches:
                        row_matches.extend(matches)
                        matched_fields[col].add((id_key, col))

                if row_matches:
                    matched_row = row.to_dict()
                    matched_row["keyword"] = "; ".join(sorted(set(row_matches)))
                    matched_rows.append(matched_row)

    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_excel("reddit_keyword_hits_byrow.xlsx", index=False)
    return matched_df

# ------------------------------
# OPTION 2: STRUCTURE-AGNOSTIC FILTERING 
# ------------------------------
def filter_by_textid():
    """
    Extracts and deduplicates text content by post, comment, and reply ID.
    Combines title + post text into a single unit.
    Matches each unique text blob against keyword regexes.
    Returns a combined DataFrame of all matched posts and comments.
    """
    post_texts = {}
    comment_texts = {}

    # Step 1: Build dictionaries of unique post/comment/reply texts
    for filename in os.listdir(input_dir):
        if filename.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(input_dir, filename))
            subreddit = filename.replace(".xlsx", "")
            for _, row in df.iterrows():
                post_id = row.get("post_id")
                comment_id = row.get("comment_id")
                reply_id = row.get("reply_id")

                # Combine title + post text
                if pd.notna(post_id) and post_id not in post_texts:
                    parts = [str(row.get("title", "")), str(row.get("post_text", ""))]
                    combined = " ".join(p for p in parts if pd.notna(p)).strip()
                    if combined:
                        post_texts[post_id] = (combined, subreddit)

                # Add comment body
                if pd.notna(comment_id) and comment_id not in comment_texts:
                    comment = str(row.get("comment_body", "")).strip()
                    if comment:
                        comment_texts[comment_id] = (comment, subreddit)

                # Add reply body as comment
                if pd.notna(reply_id) and reply_id not in comment_texts:
                    reply = str(row.get("reply_body", "")).strip()
                    if reply:
                        comment_texts[reply_id] = (reply, subreddit)

    # Step 2: Match content to keywords
    def extract_matches(text_dict, source):
        rows = []
        for uid, (text, subreddit) in text_dict.items():
            hits = [label for regex, label in regex_to_keyword.items() if re.search(regex, text, re.IGNORECASE)]
            if hits:
                rows.append({
                    "id": uid,
                    "source": source,
                    "subreddit": subreddit,
                    "text": text,
                    "keyword": "; ".join(sorted(set(hits)))
                })
        return rows

    post_matches = extract_matches(post_texts, "post")
    comment_matches = extract_matches(comment_texts, "comment")

    # Step 3: Write results to Excel with separate sheets
    with pd.ExcelWriter("reddit_filtered_by_textid.xlsx") as writer:
        pd.DataFrame(post_matches).to_excel(writer, sheet_name="Posts", index=False)
        pd.DataFrame(comment_matches).to_excel(writer, sheet_name="Comments", index=False)

    return pd.DataFrame(post_matches + comment_matches)

# ------------------------------
# MAIN PIPELINE EXECUTION
# ------------------------------

# Run the structure-agnostic filter
matched_df = filter_by_textid()

# Explode the semicolon-delimited keyword list into individual rows
df_exploded = matched_df.assign(keyword=matched_df['keyword'].str.split('; ')).explode('keyword')

# Count how often each keyword appears in each subreddit
keyword_counts = df_exploded.groupby(['subreddit', 'keyword']).size().reset_index(name='count')

# Pivot the table to show keywords as columns, subreddits as rows
keyword_summary = keyword_counts.pivot(index='subreddit', columns='keyword', values='count').fillna(0).astype(int)

# Save the summary table to Excel
keyword_summary.to_excel("keyword_summary_by_subreddit.xlsx")
