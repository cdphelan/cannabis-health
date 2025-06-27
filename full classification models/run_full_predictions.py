# remember this lives in a venv
import sqlite3
import pandas as pd
from datetime import datetime, timezone

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

import pickle, os

FILEPATH = "../data/bronze_standard_codes.csv"
OUTPATH = "../data/bronze_standard_codes_traintest.csv"
MODEL_VERSION = "v4"

# def load_gold_labels(conn):
#     # SQL query, filters out high-disagreement rows
#     query = """
#     SELECT 
#         gold.post_id,
#         gold.relevance,
#         COALESCE(p.selftext, c.body) AS text,
#         CASE 
#             WHEN p.id IS NOT NULL THEN 'post'
#             WHEN c.id IS NOT NULL THEN 'comment'
#         END AS source_type
#     FROM gold_labels AS gold
#     LEFT JOIN posts AS p ON gold.post_id = p.id
#     LEFT JOIN comments AS c ON gold.post_id = c.id
#     WHERE gold.high_disagreement = 0
#       AND (p.id IS NOT NULL OR c.id IS NOT NULL);
#     """
#     # Run the query and return results
#     return pd.read_sql_query(query, conn)

def load_gold_labels_from_file(filepath):
    # Load from CSV
    df = pd.read_csv(filepath)
    # print(df.head(20))

    # Filter out high-disagreement and kfail rows
    df = df[df["subset"] == "training_candidate"].copy()

    # Rename columns to match expectations in the rest of the script
    df = df.rename(columns={
        "source": "source_type"
    })

    return df


#tf-idf models
def run_model(gold_df, model_type="logreg", model_version="v1"):
    assert model_type in ["logreg", "xgboost", "svm"], "Unsupported model type."

    # Train-test split (for internal calibration/eval)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        gold_df["text"], gold_df["relevant"], gold_df["id"],
        test_size=0.2, random_state=42, stratify=gold_df["relevant"]
    )

    # save the training and test ids
    train_ids = set(id_train.tolist())
    write_gold_df =  pd.read_csv(FILEPATH)
    # Anything not seen during training is fair game for evaluation.
    write_gold_df["is_test"] = write_gold_df["id"].apply(lambda x: 0 if x in train_ids else 1)
    write_gold_df.to_csv(OUTPATH, index=False)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_full_vec = vectorizer.transform(gold_df["text"])

    # Initialize model
    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    elif model_type == "svm":
        model = SVC(probability=True, kernel='linear', random_state=42)
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(
                    eval_metric="logloss",
                    use_label_encoder=False,
                    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
                )

    # Train model
    model.fit(X_train_vec, y_train)

    # Predict on all gold label data (training+test)
    y_pred = model.predict(X_full_vec)
    y_proba = model.predict_proba(X_full_vec)[:, 1]

    # Construct result DataFrame
    df_results = pd.DataFrame({
        "reddit_id": gold_df["id"],
        "prediction": y_pred,
        "confidence": y_proba,
        "source_type": gold_df["source_type"],
    })
    df_results["model"] = model_type
    df_results["model_version"] = model_version
    df_results["timestamp"] = datetime.now(timezone.utc).isoformat()
    df_results["confidence_scale"] = "0-1"

    # Save model and vectorizer for reuse
    os.makedirs("models", exist_ok=True)
    with open(f"models/{model_type}_{model_version}_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(f"models/{model_type}_{model_version}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return df_results


def insert_predictions_to_db(conn, cur, df_results):
    # Create predictions table if it doesn't exist
    # THIS IS OUTDATED
    cur.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT,
            model TEXT,
            model_version TEXT,
            prediction INTEGER,
            confidence REAL,  -- usually [0, 1]
            confidence_scale TEXT,  -- e.g. 'low', 'medium', 'high'
            timestamp TEXT,
            reddit_id TEXT REFERENCES posts(id),
            PRIMARY KEY (id, model, model_version)
            )
    ''')

    for _, row in df_results.iterrows():
        cur.execute(
            '''INSERT OR REPLACE INTO predictions 
               (reddit_id, source_type, model, model_version, prediction, confidence, confidence_scale, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                row["reddit_id"], row["source_type"], row["model"],
                row["model_version"], int(row["prediction"]),
                float(row["confidence"]), row["confidence_scale"], row["timestamp"]
            )
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    conn = sqlite3.connect("reddit_combined.db")
    cur = conn.cursor()

    # Load gold standard data
    gold_df = load_gold_labels_from_file(FILEPATH)

    #tf-idf LR
    tdidf_df = run_model(gold_df, model_type="logreg", model_version=MODEL_VERSION)
    #insert_predictions_to_db(conn, cur, tdidf_df)

    #svm w tf-idf
    svm_results = run_model(gold_df, model_type="svm", model_version=MODEL_VERSION)
    #insert_predictions_to_db(conn, cur, svm_results)

    #xgboost w tf-idf
    xgb_results = run_model(gold_df, model_type="xgboost", model_version=MODEL_VERSION)
    #insert_predictions_to_db(conn, cur, xgb_results)





