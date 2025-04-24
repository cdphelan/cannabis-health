import os
import pandas as pd
import ast

# === Select & Load spaCy Model ===
import spacy
# Options: "en_core_web_sm", "en_core_sci_sm", "med7"
# open question: which one of these models is best. currently using a general purpose one 
SPACY_MODEL = "en_core_web_sm"
nlp = spacy.load(SPACY_MODEL)  


# === Helper Function: Contextual Sentence Extraction ===
def extract_cannabis_context(row, window=2):
    text = row['text']
    #a just in case redundancy, making sure 'keywords' is a lst
    keywords = ast.literal_eval(row['keywords'])
    doc = nlp(text)
    sentences = list(doc.sents)
    keyword_spans = []
    for i, sent in enumerate(sentences):
        for kw in keywords:
            if kw.lower() in sent.text.lower():
                start = max(0, i - window)
                end = min(len(sentences), i + window + 1)
                keyword_spans.append((start, end))

    # Merge overlapping spans
    merged = []
    for span in sorted(keyword_spans):
        if not merged or span[0] > merged[-1][1]:
            merged.append(span)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))

    return [" ".join([sent.text for sent in sentences[start:end]]) for start, end in merged]


def sample_rows(n, csv_folder): 
    sampled_rows = []

    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(csv_folder, filename)
            df = pd.read_csv(filepath)
            #print(df.head(3))
            sample = df.sample(n=n, random_state=42) if len(df) >= n else df
            sample['source_file'] = filename  # track origin
            for _, row in sample.iterrows():  
                mentions = extract_cannabis_context(row)
                for m in mentions:
                    sampled_rows.append({
                        "id": row.get("id"),
                        "source_file": row.get("source_file"),
                        "cannabis_context": m
                    })

    #combine & write to fil
    sampled_df = pd.DataFrame(sampled_rows)
    sampled_df.to_csv("sampled_dataset.csv", index=False)


sample_rows(50, "keyword_hits")
