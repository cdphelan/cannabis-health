NLP Pipeline for cannabis health claims on Reddit


This project extracts structured health-related claims from Reddit discussions on cannabis. It identifies compounds (e.g., THC, CBD), symptoms (e.g., pain, anxiety), and user-reported outcomes (e.g., improved sleep), along with sentiment polarity.

One of the goals of this project is to establish an analysis pipeline methodology that compares qualitative, ML, and AI methods and creates human-in-the-loop checkpoints in order to achieve a transparent and reproducible analysis pipeline for behavioral scientists.

# step1_keyword_filter.py
## Overview
This script processes Reddit data from Excel files, detects mentions of health-related keywords about
cannabis (e.g., for pain, sleep, anxiety), and summarizes them by subreddit.
It uses regex pattern matching to identify relevant terms in posts, comments, and replies.
Key Features
- Structure-agnostic filtering based on unique IDs.
- Regex-based keyword matching.
- Exports a pivot table summarizing keyword counts per subreddit.
How It Works
1. Extracts text from posts, comments, and replies.
2. Applies regex to find keyword matches.
3. Saves matched content and summary counts to Excel files.
Output Files
1. reddit_deduplicated_filtered_by_source.xlsx (Posts & Comments).
2. keyword_summary_by_subreddit.xlsx (pivoted keyword count table)


# step1_stratified_sampling.py
## Overview
This script creates stratified random samples from a keyword-tagged Reddit dataset. It supports two types of sampling:
- **By keyword**: Selects up to *n* posts per keyword.
- **By subreddit**: Selects up to *n* posts per subreddit.

The output is saved as an Excel file with two sheets, making it useful for manual annotation, content review, or reliability checks.

---

## Input
- **`reddit_keyword_hits_deduplicated.xlsx`**  
  An Excel file where each row is a Reddit post, comment, or reply that matched one or more health-related keyword patterns.  
  Assumes a column named `keyword` containing semicolon-separated keyword labels.

---

## Output
- **`reddit_fp_sampling_by_keyword_and_subreddit.xlsx`**
  - Sheet **"By Keyword"**: Random sample of up to 20 posts per keyword
  - Sheet **"By Subreddit"**: Random sample of up to 20 posts per subreddit

---

## Script Details

### `df_exploded = ...explode('keyword')`
Splits the `keyword` column into one row per keyword per post to allow keyword-level sampling.

### `stratified_sample(df, group_col, n)`
Takes a stratified sample from each group (e.g., each keyword or subreddit) of up to *n* rows.  
Uses `groupby().apply(lambda x: x.sample(...))`.

---

## Parameters

- `sample_n = 20`  
  Number of samples per group (can be adjusted).

- `seed = 42`  
  Set for reproducibility of random samples.

---

## Requirements

- Python 3
- `pandas`, `numpy`, and `openpyxl` for Excel I/O

---

## To Run

Place this script in the same directory as `reddit_keyword_hits_deduplicated.xlsx` and run:

```bash
python reddit_fp_sampling_by_keyword_and_subreddit.py
