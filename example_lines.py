import sqlite3
import pandas as pd

DB_PATH = "../../data_collection/reddit_combined.db"  # Update if needed

# Connect and read from the fulltext_examples view
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM fulltext_examples", conn)
conn.close()

# Step 1: Assign base category
def determine_base_category(row):
    if row["true_positive"]:
        return "true_positive"
    elif row["false_positive"]:
        return "false_positive"
    elif row["false_negative"]:
        return "false_negative"
    elif row["true_negative"]:
        return "true_negative"
    return None

df["base_category"] = df.apply(determine_base_category, axis=1)

# Step 2: Determine majority label per ID
majority_labels = (
    df.groupby("id")
    .agg(gold_label=("gold_label", "first"),
         disagreeing_models=("predicted_label", lambda x: (x != x.mode().iloc[0]).sum()),
         total_models=("predicted_label", "count"),
         disagreement_count=("predicted_label", lambda x: sum(x != x.name)))
    .reset_index()
)

# Join majority disagreement info back to main df
df = df.merge(majority_labels[["id", "disagreeing_models", "total_models"]], on="id", how="left")
df["disagreement_ratio"] = df["disagreeing_models"] / df["total_models"]

# Step 3: Assign "controversial" and "low_confidence"
df["category"] = df["base_category"]  # Default to base category
df.loc[df["category"].isna() & (df["disagreement_ratio"] > 0.5) & (df["predicted_label"] != df["gold_label"]),
       "category"] = "controversial"

df.loc[df["category"].isna() & (df["low_confidence"] == 1), "category"] = "low_confidence"

# Step 4: Drop rows without a predicted_label or gold_label (if not already done)
df = df[df["predicted_label"].notna() & df["gold_label"].notna()].copy()

# Step 5a: Add 10 controversial per model first
controversial = df[
    (df["disagreement_ratio"] > 0.5) &
    (df["predicted_label"] != df["gold_label"])
].copy()
controversial["category"] = "controversial"
controversial["rownum"] = controversial.groupby("model").cumcount() + 1
controversial = controversial[controversial["rownum"] <= 10]

# Step 5b: Add 10 low_confidence per model next
low_conf = df[df["low_confidence"] == 1].copy()
low_conf["category"] = "low_confidence"
low_conf["rownum"] = low_conf.groupby("model").cumcount() + 1
low_conf = low_conf[low_conf["rownum"] <= 10]

# Step 5c: Now assign base category (true/false positive/negative)
def determine_base_category(row):
    if row["true_positive"]:
        return "true_positive"
    elif row["false_positive"]:
        return "false_positive"
    elif row["false_negative"]:
        return "false_negative"
    elif row["true_negative"]:
        return "true_negative"
    return None

df["category"] = df.apply(determine_base_category, axis=1)
df_base = df[df["category"].notna()].copy()
df_base["rownum"] = df_base.groupby(["model", "category"]).cumcount() + 1
df_base = df_base[df_base["rownum"] <= 10]

# Step 5d: Combine all together
df_final = pd.concat([controversial, low_conf, df_base], ignore_index=True)
# Step 6: Final column structure
columns = [
    "model", "id", "predicted_label", "confidence", "gold_label",
    "text", "category", "low_confidence", "rownum"
]
df_final = df_final[columns]

# Step 7: Save
df_final.to_csv("data/fulltext_labeled_examples.csv", index=False)
print("Saved fulltext_labeled_examples.csv")
