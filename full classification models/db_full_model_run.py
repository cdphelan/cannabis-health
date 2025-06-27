import sqlite3
import pandas as pd
import pickle
from datetime import datetime

# === Configuration ===
DB_PATH = "../../../data_collection/reddit_combined.db"
VIEW_NAME = "cannabis_filtered_dataset"
MODELS_TO_RUN = ["logreg", "svm", "xgboost"]
MODELS_TO_RUN = ["xgboost"]

MODEL_VERSION = "v4"
DATE_NOW = datetime.now().strftime("%Y-%m-%d")

# === Connect to SQLite DB and load view ===
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(f"SELECT id, keyword FROM {VIEW_NAME}", conn)

# Ensure required columns exist
assert all(col in df.columns for col in ["id", "keyword"]), "Missing required columns"

# Prepare input text
text_input = df["keyword"].fillna("").astype(str)

# Accumulate predictions
records = []

for model_type in MODELS_TO_RUN:
    # Load vectorizer and model
    with open(f"models/{model_type}_{MODEL_VERSION}_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"models/{model_type}_{MODEL_VERSION}_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Transform text
    X = vectorizer.transform(text_input)

    # Predict class and confidence
    preds = model.predict(X)
    try:
        confidences = model.predict_proba(X).max(axis=1)
    except AttributeError:
        confidences = [None] * len(preds)

    # Add to records
    for idx, row_id in enumerate(df["id"]):
        records.append((row_id, model_type, int(preds[idx]), confidences[idx], DATE_NOW, MODEL_VERSION))

# === Write to database ===
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT,
    model TEXT,
    label INTEGER,
    confidence REAL,
    date TEXT,
    version TEXT,
    PRIMARY KEY (id, model, version)
)
""")

cur.executemany("""
INSERT OR REPLACE INTO predictions (id, model, label, confidence, date, version)
VALUES (?, ?, ?, ?, ?, ?)
""", records)

conn.commit()
conn.close()

print(f"✅ Predictions saved to 'predictions' table ({len(records)} rows).")
