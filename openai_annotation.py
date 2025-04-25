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

# Function to annotate one line with GPT
def annotate_comment(text):
    prompt = f"""
    You are an expert assistant extracting structured information from Reddit comments that describe the therapeutic use of cannabis.

    Your task is to identify and extract **each distinct mention of a cannabis dosage and compound**. For each valid mention, return a structured object with the following fields:

    - `dosage`: Convert the reported dosage into one of these bins: <10mg, 10–25mg, 25–50mg, 50–100mg, 100–500mg, 500mg+. Only include if it appears to refer to a **single-use dose** (not per container).
    - `compound`: The cannabis compound mentioned (e.g., THC, CBD, CBN, delta-8). If the dosage is attached to a general term like “weed” or “marijuana,” label this field as `"Unspecified"`.
    - `symptom`: The health condition the person is using cannabis to address. Choose from: `sleep`, `pain`, `anxiety`, or `other`.
    - `sentiment`: How effective the commenter found the compound. Choose from:
      - `positive`: It worked
      - `negative`: It did not work
      - `ambivalent`: It worked but they express reluctance or concern
      - `insufficient`: It worked but not well enough
      - `other`: None of the above clearly apply
    - `brand`: Include any brand name or strain (e.g., "Dr Solomon’s Doze Drops"). Leave as `null` if none is mentioned.

    ---

    ### Rules:
    - You may return multiple objects for a single comment—**one for each distinct compound+dosage mention**.
    - If a dosage is clearly referring to the **entire product container**, and not a single-use dose, omit that entry.
    - If the comment does **not contain any per-dose cannabis dosage mentions**, return json object with all fields null: "compound": null, "brand": null, "dosage": null, "condition": null, "sentiment": null

        Example mistake:
        Comment: "thc doesn't really put me to sleep but a high dose of cbd (150mg+) has me üò¥üí§üí§"
        Incorrect annotation: ["compound": "CBD", "brand": Null, "dosage": 150mg, "method": Null, "symptom": "sleep", "outcome": "sleep", "polarity": "insufficient"]
        Correct annotation: [("compound": "THC", "brand": Null, "dosage": Null, "method": Null, "symptom": "sleep", "outcome": "no effect", "polarity": "negative"), 
                            ("compound": "CBD", "brand": Null, "dosage": "150mg+", "method": Null, "symptom": "sleep", "outcome": "sleep improvement", "polarity": "positive")]

        Now annotate the following comment. Return only a valid JSON array on a single line. Do not include any explanations.

        Comment: \"{text.strip()}\"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",  #using a specific model ID locks in which model is using, increases replicability
            messages=[
                {"role": "system", "content": "You are a helpful assistant for medical Reddit text annotation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 #determines stochasticity, so 0.0 is no randomness, just chooses most likely next word
        )

        content = response.choices[0].message.content
        # Fix malformed JSON arrays by slicing from the first [ to the last ]
        content_clean = content[content.find("["):content.rfind("]")+1]
        parsed = json.loads(content_clean)

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError("Parsed content is not a list.")

        # Validate and add the original text to each entry
        final = []
        for entry in parsed:
            if isinstance(entry, dict):
                entry["text"] = text
                final.append(entry)
            else:
                final.append({
                    "compound": None,
                    "dosage": None,
                    "method": None,
                    "symptom": None,
                    "outcome": None,
                    "polarity": None,
                    "false_positive": True,
                    "text": text,
                    "error": "non-dict object returned"
                })
        return final

    except Exception as e:
        return [{
            "compound": None,
            "dosage": None,
            "method": None,
            "symptom": None,
            "outcome": None,
            "polarity": None,
            "false_positive": True,
            "text": text,
            "error": str(e)
        }]

def annotate_file_with_gpt(input_data, output_csv_path, rate_limit=1.5):
    import pandas as pd
    import time
    import csv

    # Load input
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    elif isinstance(input_data, str):
        df = pd.read_csv(input_data, encoding='utf-8')
    else:
        raise ValueError("input_data must be a DataFrame or a CSV file path")

    texts = df['text'].dropna().tolist()

    writer = None
    f = open(output_csv_path, mode='w', newline='', encoding='utf-8')

    for i, text in enumerate(texts):
        print(f"Annotating row {i+1}/{len(texts)}")
        rows = annotate_comment(text)

        for row in rows:
            if isinstance(row, dict):
                # Initialize writer on the first row using dynamic fieldnames
                if writer is None:
                    header = list(row.keys())
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()

                # Drop unexpected fields and fill in missing ones
                row = {key: row.get(key, None) for key in writer.fieldnames}

                writer.writerow(row)
            else:
                print(f"Skipping non-dict row: {row}")

        time.sleep(rate_limit)

    f.close()
    print(f"Finished writing: {output_csv_path}")


def main():
    input_path = "sampled_dataset.csv"
    output_path = "gpt_annotated_cannabis_data_multirow.csv"
    annotate_file_with_gpt(input_path, output_path)

if __name__ == "__main__":
    main()





