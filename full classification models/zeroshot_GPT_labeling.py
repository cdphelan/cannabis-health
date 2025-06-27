import os
from openai import OpenAI
import pandas as pd
import time
import json
import csv

# Load your gold-standard dataset
df = pd.read_csv("bronze_standard_codes.csv")  
PROMPT_TEMPLATE_PATH = "gpt_line_classification_prompt_template.txt"
OUTPUT_FILE = "gpt_zero_shot_gold-labels.csv"


from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API key
client = OpenAI(
  api_key=api_key
)

# Load template
with open(PROMPT_TEMPLATE_PATH, "r") as f:
    prompt_template = f.read()

# Resume support
if os.path.exists(OUTPUT_FILE):
    labeled_df = pd.read_csv(OUTPUT_FILE)
    completed_ids = set(labeled_df['id'])
    df = df[~df['id'].isin(completed_ids)]
    print(f"Resuming from previous run. {len(completed_ids)} lines already processed.")
    results = labeled_df
else:
    results = pd.DataFrame(columns=["id","GPT_label","GPT_confidence","GPT_reasoning"])

# Iterate through each row
for i, row in df.iterrows():
    try:
        filled_prompt = prompt_template.format(keyword=row['keyword'], text=row['text'])
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
            {"role": "system", "content": "You are a clinical NLP assistant."},
            {"role": "user", "content": filled_prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        output = response.choices[0].message.content.strip()

        print(output)
        lines = output.strip().splitlines()

        # Basic parser
        label = ""
        confidence = ""
        reasoning = ""
        for line in lines:
            if "label" in line.lower():
                label = line.split(":", 1)[-1].strip()
            elif "confidence" in line.lower():
                confidence = line.split(":", 1)[-1].strip()
            elif "reasoning" in line.lower():
                reasoning = line.split(":", 1)[-1].strip()

        row_data = {
            "id": row['id'],
            "GPT_label": label,
            "GPT_confidence": confidence,
            "GPT_reasoning": reasoning
        }

        results = pd.concat([results, pd.DataFrame([row_data])], ignore_index=True)

    except Exception as e:
        print(f"Error on row {row}: {e}")
            # Save after each row
    
    results.to_csv(OUTPUT_FILE, index=False)
    time.sleep(1.2)  # Be nice to the rate limits

