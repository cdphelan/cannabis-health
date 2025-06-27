# Re-import libraries and re-load sample data after reset

import pandas as pd

# Load previously processed sample
df_posts = pd.read_excel("reddit_filtered_by_textid.xlsx", sheet_name="Posts")
df_comments = pd.read_excel("reddit_filtered_by_textid.xlsx", sheet_name="Comments")
df_all = pd.concat([df_posts, df_comments], ignore_index=True).sample(frac=1, random_state=42)
# df_sample = df_all.head(5000).copy()
# df_sample.reset_index(drop=True, inplace=True)

# Explode keywords
df_keywords = df_all.copy()
df_keywords['keyword'] = df_keywords['keyword'].fillna('')
df_keywords = df_keywords.assign(keyword=df_keywords['keyword'].str.split(';')).explode('keyword')
df_keywords['keyword'] = df_keywords['keyword'].str.strip().str.lower()

# Frequency crosstab: count of keyword occurrences by subreddit
freq_table = pd.crosstab(df_keywords['subreddit'], df_keywords['keyword'])

# Normalize for heatmap 1: frequencies normalized by row (subreddit)
freq_by_subreddit = freq_table.div(freq_table.sum(axis=1), axis=0)

# Normalize for heatmap 2: frequencies normalized by column (keyword)
freq_by_keyword = freq_table.div(freq_table.sum(axis=0), axis=1)

# Hit rate: average relevance score per subreddit-keyword pair
# hit_table = pd.pivot_table(
#     df_keywords,
#     values='relevance',
#     index='subreddit',
#     columns='keyword',
#     aggfunc='mean'
# )

# Save results
freq_table.to_csv("keyword_frequencies.csv")
freq_by_subreddit.to_csv("freq_by_subreddit.csv")
freq_by_keyword.to_csv("freq_by_keyword.csv")
# hit_table.to_csv("hit_rate_by_subreddit_keyword.csv")

