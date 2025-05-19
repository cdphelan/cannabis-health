"""
AI Relevance Scoring Pipeline for Reddit Comments

This script uses the OpenAI API to assign relevance scores to Reddit comments discussing the therapeutic
effects of cannabis. It is designed to prioritize or pre-filter data for manual annotation or further NLP tasks.

Functionality:
- Uses GPT model to classify comment relevance:
    - 0 = No therapeutic discussion
    - 1 = Mentions therapeutic effects + specific dosage
    - 2 = Mentions therapeutic effects, but no dosage
- Designed to be lightweight and scalable
- Can be embedded in a larger filtering or triage pipeline

Requirements:
- `openai`
- `pandas`
- `dotenv` (for storing your API key in a `.env` file)

Environment Setup:
Ensure your `.env` file contains:
OPENAI_API_KEY=your_key_here
"""

import os, re
import pandas as pd
from openai import OpenAI
import time
import json

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(
  api_key=api_key
)

# === GPT Query Function ===
# This function sends a comment to the GPT model and retrieves a structured relevance score.
import os, re
import pandas as pd
from openai import OpenAI
import time
import json

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# Set your OpenAI API key
client = OpenAI(
  api_key=api_key
)

# Function to query GPT
def GPT_query_relevance(text):
    prompt = f"""
    Read this Reddit comment, which may include a discussion of the therapeutic effects of cannabis. 
    Given the text below, assign a relevance score:
    - 1: The comment contains a specific cannabis dosage (e.g., mg, ml, grams) and mentions its therapeutic effects.
    - 2: The comment discusses the therapeutic effects of cannabis, but does not include a specific dosage.
    - 0: The comment does not discuss the therapeutic effects of cannabis.

    Only return the relevance score as a number: 0, 1, or 2.

    Text:
    \"\"\"{text}\"\"\"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that strictly classifies cannabis relevance in Reddit comments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()        
        score = int(result)
        if score in [0, 1, 2]:
            return score
        else:
            return None
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

def classify_relevance(csv_folder):
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            print(f"Staring processing for {filename}")
            filepath = os.path.join(csv_folder, filename)
            df = pd.read_csv(filepath, encoding='utf-8')
            # df = df.iloc[:10]  # test on first 10 rows

            relevance_scores = []
            for i, row in df.iterrows():
                print(f"Processing {i + 1} / {len(df)}")
                score = GPT_query_relevance(row["text"])
                relevance_scores.append(score)
                time.sleep(1.5)  # rate limits

            # Write output file
            output_name = None #set to null to catch any errors in below regex
            match = re.match(r'^([^_]+_[a-zA-Z])', filename) #subreddit name + _c or _s (comments, submissions)
            if match:
                shortname = match.group(1) 
                output_name = "relevance_" + shortname + ".csv" 
            else:
                print("Error in regex: no match found: " + filename) #this shouldn't ever happen 

            # Add relevance column and export
            df["relevance"] = relevance_scores
            df.to_csv(output_name, index=False)
            print(f"Saved annotated data to {output_name}")

def compile_relevant(folder_path):
    relevant_rows = []
    for filename in os.listdir(folder_path): # Iterate over each CSV in the folder
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(filepath)
                if 'relevance' in df.columns:
                    filtered = df[df['relevance'] == 1] #pull out just the relevant ones
                    relevant_rows.append(filtered)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    compiled_df = pd.concat(relevant_rows, ignore_index=True)
    output_path = os.path.join(folder_path, "compiled_relevance_1.csv")
    compiled_df.to_csv(output_path, index=False)
    print(f"Saved {len(compiled_df)} rows with relevance == 1 to {output_path}")
    return compiled_df


def summarize_dosages(input_csv):
    # Load data
    df = pd.read_csv(input_csv)

    # Define constants
    conditions = ['sleep', 'pain', 'anxiety']
    sentiment = ['positive', 'negative', 'ambivalent', 'insufficient', 'other']
    bins = ['<10mg', '10–25mg', '25–50mg', '50–100mg', '100–500mg', '500mg+']

    # Get unique compounds
    compounds = sorted(df['compound'].dropna().unique())
    #currently hardcoded for my sanity
    compounds = ['CBC', 'CBD', 'CBD / CBN', 'CBD/THC', 'CBDA', 'CBG', 'CBG/THC', 'CBN', 
    'RSO cannabis oil', 'Sativa', 'THC', 'THC/CBD', 'THC:CBD', 'THCA', 
    'Unspecified', 'cannabis', 'cannabis oil',  'delta-8', 
    'delta-8 CBD', 'delta-8 THC', 'delta-8 or delta-9', 'delta-8/9 THC', 'delta-9', 'delta-9 THC', 
    'indica', 'indica gummy', 'med7 CBD']

    # Filter relevant rows
    df_filtered = df[
        df['dosage'].notna() &
        df['compound'].notna() &
        df['symptom'].isin(conditions) &
        df['sentiment'].isin(sentiment)
    ]

    # Initialize result container
    summary_tables = {}

    for condition in conditions:
        for s in sentiment:
            subset = df_filtered[
                (df_filtered['symptom'] == condition) &
                (df_filtered['sentiment'] == s)
            ]

            # Pivot: dosage (row) x compound (column)
            pivot = subset.pivot_table(
                index='dosage',
                columns='compound',
                aggfunc='size',
                fill_value=0
            )

            # Ensure consistent shape
            pivot = pivot.reindex(index=bins, columns=compounds, fill_value=0)

            # Remove columns with all zeros
            pivot = pivot.loc[:, (pivot != 0).any()]

            # Only keep tables with at least one non-zero cell
            if pivot.values.sum() > 0:
                summary_tables[(condition, s)] = pivot

    return summary_tables

# Example usage
if __name__ == "__main__":
    csv_path = "gpt_annotated_cannabis_data_multirow.csv"  # change to your file path



if __name__ == "__main__":
    #STEP 0: fuzzy filtered + dosage
        # do this using zst_dosages_filter() in keyword-filter-pipeline.py
    #STEP 1: zero shot semantic filtering
    # is this actually about the therapeutic effects of cannabis, and does this have a specific CANNABIS dose mentioned in it?
    # classify_relevance("keyword_hits")

    #note that in the current code you need to move the above generated csv files into a folder named relevance_scores
    # compiled_df = compile_relevant("relevance_scores") #compiles all the lines that pass the relevance filter into one file
    
    # STEP 2: entity + relation extraction    
    # from openai_annotation import annotate_file_with_gpt
    annotated_fil = "annotated_comments.csv"
    # #input_data = compiled_df
    # input_data = "relevance_scores/compiled_relevance_1.csv"
    # annotate_file_with_gpt(input_data=input_data,output_csv_path=annotated_fil)
    
    #STEP 3: summarize: mentions of binned dose, divided by compound, divided by condition, divided by sentiment
                # compound, dosage (binned), condition, polarity (binned) 
    tables = summarize_dosages(annotated_fil)
    for (condition, sentiment), table in tables.items():
        print(f"\n=== {condition.upper()} - {sentiment.upper()} ===")
        print(table)
        table.to_csv(f"{condition}_{sentiment}_summary.csv")
    #step 3.5: also summarize any brand or strain mentions


