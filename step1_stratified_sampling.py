import pandas as pd
import numpy as np

# Number of samples to draw per group
sample_n = 20

# -----------------------------------------
# Step 1: Load and Prepare Data
# -----------------------------------------

# Load the full dataset of keyword-matched Reddit entries
df = pd.read_excel("reddit_keyword_hits_deduplicated.xlsx")

# Split the 'keyword' column on semicolons into multiple rows (one keyword per row)
df_exploded = df.assign(keyword=df['keyword'].str.split('; ')).explode('keyword')

# -----------------------------------------
# Step 2: Stratified Sampling Function
# -----------------------------------------

def stratified_sample(df, group_col, n=30, seed=42):
    """
    Draws a stratified sample of up to n rows from each group in the dataframe.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_col (str): The column to group by (e.g., 'keyword' or 'subreddit').
        n (int): Maximum number of samples per group.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A new DataFrame with up to n samples per group.
    """
    np.random.seed(seed)
    return df.groupby(group_col, group_keys=False).apply(
        lambda x: x.sample(n=min(n, len(x)), random_state=seed)
    ).reset_index(drop=True)

# -----------------------------------------
# Step 3: Sample by Keyword and Subreddit
# -----------------------------------------

# Draw up to 20 samples per keyword group
sample_by_keyword = stratified_sample(df_exploded, 'keyword', n=sample_n)

# Draw up to 20 samples per subreddit group
sample_by_subreddit = stratified_sample(df_exploded, 'subreddit', n=sample_n)

# -----------------------------------------
# Step 4: Export Results
# -----------------------------------------

# Save both samples to separate sheets in a single Excel file
sample_output_path = "reddit_fp_sampling_by_keyword_and_subreddit.xlsx"
with pd.ExcelWriter(sample_output_path) as writer:
    sample_by_keyword.to_excel(writer, sheet_name="By Keyword", index=False)
    sample_by_subreddit.to_excel(writer, sheet_name="By Subreddit", index=False)