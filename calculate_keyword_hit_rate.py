
import pandas as pd

def keyword_hit_rate(input_file):
    df = pd.read_csv(input_file)

    # Normalize and explode keywords
    df['keyword'] = df['keyword'].fillna('')
    df = df.assign(keyword=df['keyword'].str.split(';')).explode('keyword')
    df['keyword'] = df['keyword'].str.strip().str.lower()

    # Calculate hit rate
    keyword_summary = df.groupby('keyword')['relevance'].agg(['count', 'mean']).rename(columns={'mean': 'hit_rate'})
    keyword_summary = keyword_summary.sort_values('hit_rate', ascending=False)

    # Save to CSV
    keyword_summary.to_csv("keyword_hit_rates.csv")

    # Also calculate hit rate by subreddit
    subreddit_summary = df.groupby('subreddit')['relevance'].agg(['count', 'mean']).rename(columns={'mean': 'hit_rate'})
    subreddit_summary = subreddit_summary.sort_values('hit_rate', ascending=False)
    subreddit_summary.to_csv("subreddit_hit_rates.csv")

    return keyword_summary, subreddit_summary

keyword_hit_rate("reddit_lines_with_gpt_relevance.csv")