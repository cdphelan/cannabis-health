# remember this lives in a venv
import sqlite3
import pandas as pd
from datetime import datetime, timezone

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

# --- Config ---
DB_PATH = "../data/reddit_cannabis.db"

# Load silver standard data
df = pd.read_csv("../silver_standard_codes.csv")
df = df[df["high_disagreement"] != 1].copy() # Filter out high-disagreement rows

#tf-idf models
def run_model(df, model_type="logreg", model_version="v1"):
    assert model_type in ["logreg", "xgboost", "svm"], "Unsupported model type."

    # Train-test split (for internal calibration/eval)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        df["text"], df["relevant"], df["id"],
        test_size=0.2, random_state=42, stratify=df["relevant"]
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_full_vec = vectorizer.transform(df["text"])

    # Initialize model
    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "svm":
        model = SVC(probability=True, kernel='linear', random_state=42)
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)

    # Train model
    model.fit(X_train_vec, y_train)

    # Predict on all data
    y_pred = model.predict(X_full_vec)
    y_proba = model.predict_proba(X_full_vec)[:, 1]

    # Construct result DataFrame
    df_results = pd.DataFrame({
        "reddit_id": df["id"],
        "source": df["source"],
        "prediction": y_pred,
        "confidence": y_proba
    })
    df_results["model"] = model_type
    df_results["model_version"] = model_version
    df_results["timestamp"] = datetime.now(timezone.utc).isoformat()
    df_results["confidence_scale"] = "0-1"

    return df_results

def insert_predictions_to_db(df_results):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Create predictions table if it doesn't exist
    cur.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        reddit_id TEXT,
        source_type TEXT,
        model TEXT,
        model_version TEXT,
        prediction INTEGER,
        confidence REAL,
        confidence_scale TEXT,
        timestamp TEXT,
        PRIMARY KEY (reddit_id, model, model_version)
    )
    ''')

    for _, row in df_results.iterrows():
        cur.execute(
            '''INSERT OR REPLACE INTO predictions 
               (reddit_id, source_type, model, model_version, prediction, confidence, confidence_scale, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                row["reddit_id"], row["source"], row["model"],
                row["model_version"], int(row["prediction"]),
                float(row["confidence"]), row["confidence_scale"], row["timestamp"]
            )
        )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    #tf-idf LR
    tdidf_df = run_model(df, model_type="logreg", model_version="v1")
    insert_predictions_to_db(tdidf_df)

    #xgboost w tf-idf
    svm_results = run_model(df, model_type="svm", model_version="v1")
    insert_predictions_to_db(svm_results)

    #xgboost w tf-idf
    xgb_results = run_model(df, model_type="xgboost", model_version="v1")
    insert_predictions_to_db(xgb_results)





