#compare between two coders' labeled samples

import pandas as pd
from sklearn.metrics import cohen_kappa_score

def compare_csv_labels(csv1_path, csv2_path):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    df2.rename(columns={"relevance": "label"}, inplace=True) # for the GPT-labeled data, column name is wrong

    inconsistent_rows = {'file1': [], 'file2': []}

    # Create matching key
    df1['match_key'] = df1['id'].astype(str) + "_" + df1['subreddit'].astype(str)
    df2['match_key'] = df2['id'].astype(str) + "_" + df2['subreddit'].astype(str)

    # Internal consistency checks
    for df, name in [(df1, 'file1'), (df2, 'file2')]:
        dup = df[df.duplicated('match_key', keep=False)]
        grouped = dup.groupby('match_key')['label']
        for key, labels in grouped:
            if labels.nunique() > 1:
                lines = df.index[df['match_key'] == key].tolist()
                inconsistent_rows[name].append((key, lines))

    if inconsistent_rows['file1'] or inconsistent_rows['file2']:
        print("❗ Inconsistent labeling within files:")
        for fname, errors in inconsistent_rows.items():
            for key, lines in errors:
                print(f"{fname} - key: {key}, line(s): {lines}")
        return None

    # Match across files using match_key
    merged = pd.merge(df1[['match_key', 'label']], df2[['match_key', 'label']],
                      on='match_key', suffixes=('_1', '_2'))

    total_matches = len(merged)
    agreement = (merged['label_1'] == merged['label_2']).sum()
    agreement_rate = agreement / total_matches if total_matches > 0 else 0.0

    print(f"Total matching keys (id + subreddit): {total_matches}")
    print(f"Matching labels: {agreement}")
    print(f"Agreement rate: {agreement_rate:.2%}")
    print(f"Cohen's Kappa: {cohen_kappa_score(merged['label_1'], merged['label_2']):.3f}")

    # --- Keyword Analysis and Export ---

    def explode_keywords(df, with_label=False):
        df_kw = df[['match_key', 'keyword']].copy() if not with_label else df[['match_key', 'keyword', 'label']].copy()
        df_kw['keyword'] = df_kw['keyword'].fillna('')
        df_kw = df_kw.assign(keyword=df_kw['keyword'].str.split(';')).explode('keyword')
        df_kw['keyword'] = df_kw['keyword'].str.strip().str.lower()
        return df_kw.drop_duplicates() if not with_label else df_kw

    # Hit rate analysis
    def keyword_hit_rates(df):
        kw_df = explode_keywords(df, with_label=True)
        return kw_df.groupby('keyword')['label'].agg(['count', 'mean']).rename(columns={'mean': 'hit_rate'})

    rate1 = keyword_hit_rates(df1)
    rate2 = keyword_hit_rates(df2)

    comparison = rate1.join(rate2, lsuffix='_1', rsuffix='_2', how='outer').fillna(0)
    comparison['hit_rate_diff'] = (comparison['hit_rate_1'] - comparison['hit_rate_2']).abs()
    comparison_sorted = comparison.sort_values('hit_rate_diff', ascending=False)

    print("\nKeyword hit rate differences (most disagreement first):")
    print(comparison_sorted[['hit_rate_1', 'hit_rate_2', 'hit_rate_diff']].head(20))

    # Shared keyword lines
    keywords1 = explode_keywords(df1)
    keywords2 = explode_keywords(df2)
    shared_counts = pd.merge(keywords1, keywords2, on=['match_key', 'keyword'])['keyword'].value_counts().rename('n_shared')

    print("\nKeyword shared coding counts (lines both coders labeled):")
    print(shared_counts.head(20))

    # Export
    comparison_sorted.to_csv("keyword_hit_rate_comparison.csv")
    shared_counts.to_frame().to_csv("keyword_shared_counts.csv")
    print("\nCSV export complete: 'keyword_hit_rate_comparison.csv' and 'keyword_shared_counts.csv'")
    def keyword_hit_rates(df):
        kw_df = df.copy()
        kw_df['keyword'] = kw_df['keyword'].fillna('')
        kw_df = kw_df.assign(keyword=kw_df['keyword'].str.split(';')).explode('keyword')
        kw_df['keyword'] = kw_df['keyword'].str.strip().str.lower()
        summary = kw_df.groupby('keyword')['label'].agg(['count', 'mean']).rename(columns={'mean': 'hit_rate'})
        return summary

    rate1 = keyword_hit_rates(df1)
    rate2 = keyword_hit_rates(df2)

    comparison = rate1.join(rate2, lsuffix='_1', rsuffix='_2', how='outer').fillna(0)
    comparison['hit_rate_diff'] = (comparison['hit_rate_1'] - comparison['hit_rate_2']).abs()
    comparison_sorted = comparison.sort_values('hit_rate_diff', ascending=False)

    print("\nKeyword hit rate differences (most disagreement first):")
    print(comparison_sorted[['hit_rate_1', 'hit_rate_2', 'hit_rate_diff']].head(20))

    # Calculate number of lines both coders labeled, by keyword
    print("\nKeyword shared coding counts (lines both coders labeled):")

    def prepare_keywords(df):
        exploded = df[['match_key', 'keyword']].copy()
        exploded['keyword'] = exploded['keyword'].fillna('')
        exploded = exploded.assign(keyword=exploded['keyword'].str.split(';')).explode('keyword')
        exploded['keyword'] = exploded['keyword'].str.strip().str.lower()
        return exploded.drop_duplicates()

    keywords1 = prepare_keywords(df1)
    keywords2 = prepare_keywords(df2)

    shared_keywords = pd.merge(keywords1, keywords2, on=['match_key', 'keyword'])
    shared_counts = shared_keywords['keyword'].value_counts().rename('n_shared')
    print(shared_counts.head(20))

    return agreement_rate


compare_csv_labels("reddit_labeled_output.csv", "reddit_lines_with_gpt_relevance.csv")

