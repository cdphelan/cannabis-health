import os
from openai import OpenAI
import pandas as pd
import time
import json
import csv

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API key
client = OpenAI(
  api_key=api_key
)

# Setup
INPUT_FILE = "reddit_filtered_by_textid.xlsx"
PROMPT_TEMPLATE_PATH = "gpt_line_classification_prompt_template.txt"
OUTPUT_FILE = "reddit_lines_with_gpt_relevance.csv"

# Load template
with open(PROMPT_TEMPLATE_PATH, "r") as f:
    prompt_template = f.read()

# Load data
posts = pd.read_excel(INPUT_FILE, sheet_name="Posts")
comments = pd.read_excel(INPUT_FILE, sheet_name="Comments")
df = pd.concat([posts, comments], ignore_index=True)

# Resume support
if os.path.exists(OUTPUT_FILE):
    labeled_df = pd.read_csv(OUTPUT_FILE)
    completed_ids = set(labeled_df['id'])
    df = df[~df['id'].isin(completed_ids)]
    print(f"Resuming from previous run. {len(completed_ids)} lines already processed.")
    results = labeled_df
else:
    results = pd.DataFrame(columns=df.columns.tolist() + ['relevance'])

# Iterate through each row
for i, row in df.iterrows():
    try:
        filled_prompt = prompt_template.format(keyword=row['keyword'], text=row['text'])
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[{"role": "user", "content": filled_prompt}],
            temperature=0,
            max_tokens=10
        )
        relevance = response.choices[0].message.content.strip() 
        if relevance not in ["0", "1"]:
            print(f"Unexpected output for ID {row['id']}: {relevance}")
            relevance = ""

        row_data = row.to_dict()
        row_data['relevance'] = relevance
        results = pd.concat([results, pd.DataFrame([row_data])], ignore_index=True)

        # Save after each row
        results.to_csv(OUTPUT_FILE, index=False)
        time.sleep(1.5)  # polite delay to avoid rate limit

    except Exception as e:
        print(f"Error on row {i} (ID {row['id']}): {e}")
        break  # optional: pause on error for manual review

print("✅ Classification complete. Output saved to", OUTPUT_FILE)
