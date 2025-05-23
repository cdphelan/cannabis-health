import pandas as pd
import numpy as np

# Number of samples to draw per group
sample_n = 10

#set subset of keywords to sample
keyword_subset = None
keyword_subset = ["always tired","anxiety","can't fall asleep","chest tightness","discomfort","feels like i'm dying","flare-up","hypersomnia","inflammation","insomnia","melatonin","migraine","nightmares","no energy","on edge","oversleep","pain","panic","paranoia","racing heart","shortness of breath","sleep","slept","snooze","soreness","sweating","throbbing","tired all the time","tossing and turning","wake up at night","wide awake","ache","anxious","bedtime","dream","exhausted","freaking out","headache","heart pounding","nervous","overwhelmed","spiraling","stinging"]

already_labeled = None
already_labeled = ["joc8z4g","lrvuznf","1eyrmh4","ji238zs","li4a8dt","1fxsgi9","lscy1om","gsvll23","lpuxnwd","ljyqxnd","10on5ay","1gfo94o","jl3ah5c","ldi7hku","ka1gznz","lgssxjw","licxiw7","lqtqqv6","ljpqcs8","lvkqq1i","lbuz60e","lntqkmt","lj3kf5g","u12n90","lohhtbx","jc7sh0h","hiy12s1","ig3q4aw","1f068te","j0rc1ra","i3uqntj","lpoux3r","lvcuoum","lu2gu0h","lu2zoo8","lk5a2qx","lkah1dy","lnlze0i","ljwcn6q","lwt5jo3","li8m04f","1f2phej","1g9a0p5","129s7sj","lwgk226","kbgs8e2","lldt7ws","lx0wh1d","1f141n3","lb5lo53","lbmcg4m","llk1br4","1f2go8f","loj96ej","lneeh46","lbkx1gg","1ez69u0","l9r9m5s","1f06zir","y2m3wf","1elwayw","12mq2t9","lum2oy1","lgnafs1","rsz0uw","lux6j2t","lk1qjkq","lr0t4oo","lrsdq1b","lvjyvme","lsuqhhg","lketdmk","lqk7xoe","j879xu3","lqe2gf4","ljgjs2k","ljp9tma","1dsjnrp","lk4xz35","1f3h0zd","1ft86py","lxakuac","leoejkg","lfzytpu","lq3ixaj","lj948a6","l7q65dh","lk63ik4","1gjt55c","lmon659","ltocbc0","1ev1han","itsmphb","lvovftk","lk7c84u","10qqmr9","h14qma2","lj9yrxy","1f19o2h","lsitok4","j8t7qco","lweu57i","lm7lu57","lx0vm70","ll0g8bt","lidarq0"]
# -----------------------------------------
# Step 1: Load and Prepare Data
# -----------------------------------------

# Load the full dataset of keyword-matched Reddit entries
posts = pd.read_excel("reddit_filtered_by_textid.xlsx", sheet_name="Posts")
comments = pd.read_excel("reddit_filtered_by_textid.xlsx", sheet_name="Comments")
df = pd.concat([posts, comments], ignore_index=True)

# Split the 'keyword' column on semicolons into multiple rows (one keyword per row)
df_exploded = df.assign(keyword=df['keyword'].str.split('; ')).explode('keyword')
df_exploded['keyword'] = df_exploded['keyword'].str.strip().str.lower()
df_exploded = df_exploded.drop_duplicates()

# Filter to only rows with selected keywords
if keyword_subset:
    df_exploded = df_exploded[df_exploded["keyword"].str.lower().isin([kw.lower() for kw in keyword_subset])]

if already_labeled:
    df_exploded = df_exploded[~df_exploded['id'].isin(already_labeled)]

# Create a mapping from id to all matched keywords
keyword_map = df_exploded.groupby('id')['keyword'].apply(lambda kws: '; '.join(sorted(set(kws)))).to_dict()

# Remove duplicates across keywords and assign full keyword list
unique_rows = df.drop_duplicates(subset=['id'])
unique_rows['keyword'] = unique_rows['id'].map(keyword_map)

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
    #np.random.seed(seed) #toggle this on or off if you want it to be a stable sampling
    seen_ids = set()
    samples = []

    for group_val, group_df in df.groupby(group_col):
        group_df = group_df[~group_df['id'].isin(seen_ids)]
        sample = group_df.sample(n=min(n, len(group_df)))
        samples.append(sample)
        seen_ids.update(sample['id'])

    return pd.concat(samples).reset_index(drop=True)

# -----------------------------------------
# Step 3: Sample by Keyword and Subreddit
# -----------------------------------------

# Draw up to n samples per keyword group
sample_by_keyword = stratified_sample(df_exploded, 'keyword', n=sample_n)

# Draw up to n samples per subreddit group
sample_by_subreddit = stratified_sample(df_exploded, 'subreddit', n=sample_n)

# -----------------------------------------
# Step 4: Export Results
# -----------------------------------------

# Save both samples to separate sheets in a single Excel file
sample_output_path = "reddit_sample.xlsx"
with pd.ExcelWriter(sample_output_path) as writer:
    sample_by_keyword.to_excel(writer, sheet_name="By Keyword", index=False)
    sample_by_subreddit.to_excel(writer, sheet_name="By Subreddit", index=False)