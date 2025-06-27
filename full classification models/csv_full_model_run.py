import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Load input dataset
df = pd.read_csv("../data/all_cannabis-health_622.csv")

# Check columns
assert all(col in df.columns for col in ["id", "subreddit", "keyword"]), "Missing required columns"

# Use the keyword field as the text to classify
text_input = df["keyword"].fillna("").astype(str)

# Models to apply
models_to_run = ["logreg", "svm", "xgboost"]
model_version = "v1"

# Accumulate results
results = []

for model_type in models_to_run:
    # Load vectorizer and model
    with open(f"models/{model_type}_{model_version}_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"models/{model_type}_{model_version}_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Transform text
    X = vectorizer.transform(text_input)

    # Predict
    predictions = model.predict(X)

    # Store results
    model_df = df.copy()
    model_df["model"] = model_type
    model_df["relevance"] = predictions
    results.append(model_df[["id", "subreddit", "keyword", "model", "relevance"]])

# Combine all model results
combined_results = pd.concat(results, ignore_index=True)

# Save to CSV
combined_results.to_csv("all_cannabis-health_with_predictions.csv", index=False)

print(f"✅ Done. Output saved to all_cannabis-health_with_predictions.csv with {len(combined_results)} rows.")