#this is sloppily adapted which is why some of the variable names are weird
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os, time

# Load files
INPUT_FILE = "../silver_standard_codes.csv"
OUTPUT_FILE = "gpt_few_shot_gold-labels.csv"
MODEL_NAME = "gpt-4.1-nano-2025-04-14"
SLEEP_SECONDS = 1.2

# Load data
df = pd.read_csv(INPUT_FILE)
gold_df = df.dropna(subset=["relevant"])

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API key
client = OpenAI(
  api_key=api_key
)

# Construct few-shot examples from gold standard
few_shot_examples = ""
for _, row in gold_df.sample(n=min(7, len(gold_df)), random_state=42).iterrows():
    few_shot_examples += f"Text: {row['text']}\nRelevance: {row['relevant']}\n\n"

# Define the prompt template
def build_prompt(text):
    return f"""You are an expert in analyzing Reddit posts for discussions of therapeutic cannabis use.

## Relevance Definition
A post is considered **relevant** if it clearly discusses cannabis in relation to any of the following three health concerns:
- Pain
- Anxiety
- Sleep

**Mark the post as relevant (Label: 1) if it includes:**
- Mentions of cannabis (including derivatives like CBD, CBN, delta-8, etc.) being used to treat, manage, or affecting the symptoms of these conditions
- Descriptions of cannabis improving or worsening symptoms
- Mentions of symptoms caused by cannabis use or by cannabis withdrawal

**Do NOT mark the post as relevant (Label: 0) if:**
- It only discusses other health effects unrelated to pain, anxiety, or sleep (e.g., appetite, nausea, focus)
- The connection between cannabis and one of the target conditions is vague or missing

**Common edge cases:**
- Headaches count as a pain symptom
- Being "calmed" or "overwhelmed" by cannabis counts as affecting anxiety
- Changes to dreams or dream recall count as sleep-related
- Posts must mention both cannabis and at least one of the three target conditions to be relevant

## Few-shot labeled examples

{few_shot_examples}

---

Now evaluate the following Reddit post:

Text: {text}

Respond in this format:
Label: [1 / 0]
Confidence: [Low / Medium / High]
Reasoning: [One-sentence explanation]
""".strip()


# Get set of already processed IDs (for restartability)
if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE)
    processed_ids = set(existing["id"].astype(str))
else:
    with open(OUTPUT_FILE, "w") as f:
        f.write("id,GPT_label,GPT_confidence,GPT_reasoning\n")
    processed_ids = set()

# Loop through remaining items
for i, row in tqdm(df.iterrows(), total=len(df)):
    if str(row["id"]) in processed_ids:
        continue  # Skip already processed

    try:
        prompt = build_prompt(row["text"])
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a clinical NLP assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )

        output = response.choices[0].message.content.strip()
        lines = output.splitlines()

        label, confidence, reasoning = "", "", ""
        for line in lines:
            if "label" in line.lower():
                label = line.split(":", 1)[-1].strip()
            elif "confidence" in line.lower():
                confidence = line.split(":", 1)[-1].strip()
            elif "reasoning" in line.lower():
                reasoning = line.split(":", 1)[-1].strip()

        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{row['id']},{label},{confidence},\"{reasoning.replace(',', ';')}\"\n")

    except Exception as e:
        print(f"Error on row {row['id']}: {e}")

    time.sleep(SLEEP_SECONDS)
