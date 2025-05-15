import pandas as pd
import os
import re

from keywords_pain_sleep_anxiety import pain_sleep_anxiety as regex_to_keyword

# Define the directory containing the .xlsx files
input_dir = "../Raw"  # Placeholder path


regex_patterns = list(regex_to_keyword.keys())
text_columns = ["title", "post_text", "comment_body", "reply_body"]

# Trackers to avoid duplicate matches
matched_fields = {
    "post_text": set(),
    "title": set(),
    "comment_body": set(),
    "reply_body": set()
}

matched_rows = []

# Iterate over all files in the directory
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

# Save results
matched_df = pd.DataFrame(matched_rows)
output_path = "reddit_keyword_hits_deduplicated.xlsx"
matched_df.to_excel(output_path, index=False)

output_path