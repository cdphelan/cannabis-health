import pandas as pd
import numpy as np

sample_n = 20

# Create a copy of the data and explode keywords for sampling
df = pd.read_excel("reddit_keyword_hits_deduplicated.xlsx")
df_exploded = df.assign(keyword=df['keyword'].str.split('; ')).explode('keyword')

# Function to stratified sample up to n samples per group
def stratified_sample(df, group_col, n=30, seed=42):
    np.random.seed(seed)
    return df.groupby(group_col, group_keys=False).apply(
                lambda x: x.sample(n=min(n, len(x)), random_state=seed)
                ).reset_index(drop=True)

# Sample 30 rows per keyword
sample_by_keyword = stratified_sample(df_exploded, 'keyword', n=sample_n)

# Sample 30 rows per subreddit
sample_by_subreddit = stratified_sample(df_exploded, 'subreddit', n=sample_n)

# Save both to separate sheets in the same Excel file
sample_output_path = "reddit_fp_sampling_by_keyword_and_subreddit.xlsx"
with pd.ExcelWriter(sample_output_path) as writer:
    sample_by_keyword.to_excel(writer, sheet_name="By Keyword", index=False)
    sample_by_subreddit.to_excel(writer, sheet_name="By Subreddit", index=False)
