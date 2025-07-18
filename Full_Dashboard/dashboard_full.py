import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import requests
import altair as alt
import re
import json


# === CONFIG === #
# Hidden file ID (you could also store this in st.secrets)
# FILE_ID = st.secrets["gdrive"]["file_id"]
# WHEN IN LOCAL - all these ("Full_Dashboard/data/) need to be changed to these ("data/)
FULL_DATA_PATH = "Full_Dashboard/data/dashboard_csv_data.csv"
GOLD_LABEL_PATH = "Full_Dashboard/data/bronze_standard_codes_traintest.csv"

st.set_page_config(layout="wide")

# Override default theme temporarily
st.markdown("""
    <style>
        body {
            color: #000000;
            background-color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)

# === INTRO DROP DOWN ====
# Initialize toggle in session state
if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

def toggle_intro():
    st.session_state.show_intro = not st.session_state.show_intro

# Style for full-width collapsible panel
st.markdown("""
    <style>
    .intro-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-bottom: 2px solid #ccc;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        border-radius: 0 0 12px 12px;
        margin-bottom: 1rem;
    }
    .toggle-button {
        font-size: 20px;
        color: #555;
        margin-bottom: 10px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)




# === END INTRO DROP DOWN ===

# CATEGORY_COLORS reused from original interface
CATEGORY_COLORS = {
    "compound": "#FF6B6B",
    "dosage": "#F4A261",
    "ingestion_method": "#2A9D8F",
    "use_context": "#E9C46A",
    "valence": "#9B5DE5",
    "symptom_keyword": "#00B4D8",
    "mapped_condition": "#6A4C93",
    "side_effects": "#B5838D",
    "mechanism": "#264653",
    "certainty": "#1D3557",
    "needs_review": "#FFB703",
    "review_reason": "#FB8500"
}


# === LOAD DATA === #
# TODO(dev): Handle missing relevance values more explicitly before production
# TODO(dev): better handling of the keyword_matches table, which currently needs to be manually refreshed with populate_keyword_matches.py
# @st.cache_data
def load_data_dual():
    # try:
    df_full = pd.read_csv(FULL_DATA_PATH)
    # except:
    #     download_url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    #     response = requests.get(download_url)
    #     if response.status_code != 200:
    #         st.error(f"Failed to download data: {response.status_code}")
    #         return pd.DataFrame(), pd.DataFrame()
    #     csv_content = io.StringIO(response.text)
    #     df_full = pd.read_csv(csv_content)

    df_full["keyword"] = df_full["keyword"].fillna("")
    # df_full = df_full[(df_full["keyword"] != "") | (df_full["subset"] == "random_kfails")] #cut the random lines with no keyword matches - I think they're version conflicts where a comment was deleted

    # Base (non-exploded) version for gold label evaluations
    df_base = df_full.copy()

    # Exploded version for keyword diagnostics
    df_keywords = df_full.assign(keyword=df_full["keyword"].str.split(",")).explode("keyword")
    df_keywords["keyword"] = df_keywords["keyword"].str.strip().str.lower()

    return df_keywords, df_base

#hand over data (filtered on model + keyword fails)
def apply_filters(df, model, exclude_kfails=True):
    df = df[df["model"] == model]
    if exclude_kfails:
        df = df[df["subset"] != "random_kfails"]
    return df

def filter_dataset(df, keywords, subreddits):
    return df[df["keyword"].str.lower().isin(keywords) & df["subreddit"].isin(subreddits)]

# ------------------------- MAIN -------------------------
# load data: df_base (original), df_keywords (keyword exploded), gold_df (gold labels)
df_keywords, df_base = load_data_dual()
gold_df = pd.read_csv(GOLD_LABEL_PATH)

# Sidebar page navigation
page = st.sidebar.radio("Select a Page", ["💡 Introduction", "📈 Diagnostics", "📄 Sample Labels", "📝 Sample Annotations"])

#select which model to use
models = df_base["model"].dropna().astype(str).unique()
selected_model = st.sidebar.selectbox("Select learning model:", sorted(models))

#redundant for Tab1, creates a standard global dataset. only if needed for other tabs
# df_base = apply_filters(df_base, selected_model, exclude_kfails=True)

if page == "💡 Introduction":
    st.header("How Do Inputs Change Outputs?")
    st.markdown("_An interactive NLP pipeline to explore how small decisions shape big model outcomes._")

    st.write(
        """This dashboard compares how classical machine learning and GenAI models perform on a single task: 
        finding health-related cannabis discussions on Reddit. Explore how changes in filters, prompts, and models affect 
        what gets labeled, and what gets missed."""
    )

    st.markdown(
        '<span style="color:red;">**If you are at RSMj**, feel free to track down Chanda for a demo of the dashboard! She is also interested in developing this tool out for other projects.',
        unsafe_allow_html=True,
    )
    
    st.markdown("### How to get started:")
    st.image("Full_Dashboard/img/page_tabs.png", width=300, caption="Flip through different tabs to explore.")
    st.markdown("**To explore how keywords and subreddits affect different models:**")
    st.markdown(
        """
1. Make sure you are on the 📈 Diagnostics tab (top left sidebar).  
2. Select which model you want to use (dropdown menu in left sidebar).  
3. Check the model's performance against the gold standard data, labeled by humans (top of main pane).  
4. Use the sliders (left sidebar) to exclude different keywords and subreddits and see if your decisions change the performance.  
"""
    )

    st.markdown("**To explore the decisions models make about whether to include a post for more analysis:**")
    st.markdown("Check out 📄 Sample Labels, which shows real-text examples and the models' decisions about it.")

    st.markdown("**To explore how GPT-4.1 Nano performed in an annotation task:**")
    st.markdown("Check out 📝 Sample Annotations, which shows annotations of real-text examples.")

    st.markdown(
        '<span style="color:red; font-family:Courier;"><strong>NOTE:</strong> This is a prototype for <strong>DEMONSTRATION PURPOSES</strong>. Outcomes are not verified.</span>',
        unsafe_allow_html=True,
    )


    st.markdown(
        "_First author & developer: Dr. Chanda Phelan Kane. Part of the CannTalk project by Dr. Kristina Jackson (Rutgers U) and Dr. Jane Metrik (Brown U)._"
    )


if page == "📈 Diagnostics":
    # st.header("🧪 Diagnostic View")

    #toggle the keyword filter fails on and off
    exclude_kfails = st.sidebar.checkbox("Turn on keyword filter layer", value=True)
 
    # apply first layer of filters: model & kfails
    df_keywords = apply_filters(df_keywords, selected_model, exclude_kfails)
    df_base_all = apply_filters(df_base, selected_model, exclude_kfails)
    # print(df_base[df_base["id"] == "1ao0gdt"]) #not empty

    # Filters
    st.sidebar.subheader("🔧 Threshold Filters")

    min_kw_freq = st.sidebar.number_input("Min Keyword Frequency", min_value=0, value=0)
    min_kw_hit = st.sidebar.slider("Min Keyword Hit Rate", 0.0, 1.0, 0.0)
    min_kw_score = st.sidebar.slider("Min Keyword Usefulness", 0.0, 10.0, 0.0)

    min_sub_freq = st.sidebar.number_input("Min Subreddit Frequency", min_value=0, value=0)
    min_sub_hit = st.sidebar.slider("Min Subreddit Hit Rate", 0.0, 1.0, 0.0)
    min_sub_score = st.sidebar.slider("Min Subreddit Usefulness", 0.0, 10.0, 0.0)


    # Subreddit filtering
    subreddit_summary = df_keywords.groupby("subreddit")["relevance"].agg(["count", "mean"])
    subreddit_summary.columns = ["frequency", "hit_rate"]
    subreddit_summary["usefulness_score"] = subreddit_summary["hit_rate"] * np.log(subreddit_summary["frequency"] + 1)

    #list of subs after filters
    filtered_subs = subreddit_summary[
        (subreddit_summary["frequency"] >= min_sub_freq) &
        (subreddit_summary["hit_rate"] >= min_sub_hit) &
        (subreddit_summary["usefulness_score"] >= min_sub_score)
    ].index.tolist()

    df_keywords_sub = df_keywords[df_keywords["subreddit"].isin(filtered_subs)] #filter down the data according to subreddit filters

    # Keyword filtering
    keyword_summary = df_keywords_sub.groupby("keyword")["relevance"].agg(["count", "mean"])
    keyword_summary.columns = ["frequency", "hit_rate"]
    keyword_summary["usefulness_score"] = keyword_summary["hit_rate"] * np.log(keyword_summary["frequency"] + 1)

    #list of keywords after filters
    filtered_keywords = keyword_summary[
        (keyword_summary["frequency"] >= min_kw_freq) &
        (keyword_summary["hit_rate"] >= min_kw_hit) &
        (keyword_summary["usefulness_score"] >= min_kw_score)
    ].index.tolist()

    df_keywords_filtered = df_keywords_sub[df_keywords_sub["keyword"].isin(filtered_keywords)] #filter down the data according to keyword & subreddit filters
    # print(df_keywords_filtered.head(20))

    # NEW: filter IDs and use them to split df_base
    matching_ids = set(df_keywords_filtered["id"].unique())
    df_base_included = df_base_all[df_base_all["id"].isin(matching_ids)].copy()
    df_base_excluded = df_base_all[~df_base_all["id"].isin(matching_ids)].copy()

    # Mark filtered-out examples as predicted not relevant
    df_base_excluded["relevance"] = 0

    # Combine both for final evaluation set
    # df_base = df_base[df_base["id"].isin(df_keywords_filtered["id"].unique())] # this line is mutually exclusive with below. this line drops all filtered-out lines, below sets them as 0
    df_base = pd.concat([df_base_included, df_base_excluded], ignore_index=True)
    df_base = df_base.drop_duplicates("id") #this should be redundant
    
    # Diagnostic Metrics
    # st.subheader("🧪 Diagnostic Metrics")

    # renaming some columns for prep
    gold_df = gold_df.rename(columns={"relevant": "relevance_true"})
    df_eval = df_base.rename(columns={"relevance": "relevance_pred"}).copy()


    # Branch: classical vs GPT model (affects which gold labels are used in the eval)
    if selected_model.lower() in ["logreg", "svm", "xgboost"]:
        gold_eval = gold_df[gold_df["is_test"] == 1]
        #gold_eval = gold_eval[gold_eval["subset"] == "training_candidate"] # gets rid of the high disagreement lines that might be harder to code right
    else:
        gold_eval = gold_df  # use full gold label set for GPT and other models


    # Now merge only with appropriate gold labels
    df_eval_filtered = df_eval.merge(gold_eval, on="id")
    # print(df_eval_filtered.head(20))
    # print(df_eval_filtered['id'])
    # print((df_eval_filtered["relevance_true"] == 1).sum())
    print(len(df_base_excluded.merge(gold_eval, on="id")))

    true_positives = ((df_eval_filtered["relevance_pred"] == 1) & (df_eval_filtered["relevance_true"] == 1)).sum()
    predicted_positives = (df_eval_filtered["relevance_pred"] == 1).sum()
    actual_positives = (df_eval_filtered["relevance_true"] == 1).sum()
    correct_preds = (df_eval_filtered["relevance_pred"] == df_eval_filtered["relevance_true"]).sum()

    precision = true_positives / predicted_positives if predicted_positives else 0
    recall = true_positives / actual_positives if actual_positives else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    total = len(df_eval)
    relevant = df_eval["relevance_pred"].sum()

    st.markdown("### ✅ Evaluation Against Gold Labels")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🔍 Evaluation Rows", len(df_eval_filtered))
    col2.metric("✅ Correct Predictions", correct_preds)
    col3.metric("🎯 Precision (Gold)", f"{precision:.2%}")
    col4.metric("📈 Recall (Gold)", f"{recall:.2%}")
    col5.metric("📊 F1 Score (Gold)", f"{f1:.2f}")

    st.markdown("### Full Dataset Metrics")
    col_a1, col_a2, col_a3 = st.columns(3)
    col_a1.metric("📦 Filtered Rows", total)
    col_a2.metric("🤖 Predicted Relevant", int(relevant))
    col_a3.metric("🔍 From Keyword Match to Semantic Relevance", f"{int(relevant) / total:.2%}")

    # Usefulness Tables
    st.markdown("### 🔑 Keyword and Subreddit Usefulness")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Keyword Usefulness**")
        st.dataframe(keyword_summary.loc[filtered_keywords].sort_values("usefulness_score", ascending=False).reset_index(),
                     use_container_width=True)
    with col2:
        st.write("**Subreddit Usefulness**")
        st.dataframe(subreddit_summary.loc[filtered_subs].sort_values("usefulness_score", ascending=False).reset_index(),
                     use_container_width=True)

    # Grouped Bar Chart for Keywords
    st.markdown("### 📊 Keyword Frequency by Relevance")

    sorted_keywords = keyword_summary.loc[filtered_keywords].sort_values("frequency", ascending=False).copy()
    sorted_keywords["relevant"] = (sorted_keywords["frequency"] * sorted_keywords["hit_rate"]).round().astype(int)
    sorted_keywords["irrelevant"] = (sorted_keywords["frequency"] - sorted_keywords["relevant"]).astype(int)
    sorted_keywords = sorted_keywords.reset_index()

    keyword_freq_melted = sorted_keywords.melt(
        id_vars="keyword",
        value_vars=["relevant", "irrelevant"],
        var_name="label",
        value_name="count"
    )

    keyword_chart = alt.Chart(keyword_freq_melted).mark_bar().encode(
        x=alt.X('keyword:N', title='Keyword'),
        y=alt.Y('count:Q', title='Frequency'),
        color='label:N',
        tooltip=['keyword', 'label', 'count']
    ).properties(width=20 * len(sorted_keywords), height=400)

    from streamlit.components.v1 import html
    chart_html = keyword_chart.to_html()
    html(f"""
    <div style="overflow-x:auto; border:1px solid #ccc; padding:1rem;">
        {chart_html}
    </div>
    """, height=500)

    # === Grouped Bar Chart: Subreddit Distribution for Selected Keyword ===
    st.markdown("### 🔍 Subreddit Distribution for Selected Keyword")

    keyword_subreddit_summary = (
        df_keywords_filtered
        .groupby(["keyword", "subreddit"])["relevance"]
        .agg(["count", "mean"])
        .rename(columns={"count": "frequency", "mean": "hit_rate"})
        .reset_index()
    )
    keyword_subreddit_summary["relevant"] = (keyword_subreddit_summary["frequency"] * keyword_subreddit_summary["hit_rate"]).round().astype(int)
    keyword_subreddit_summary["irrelevant"] = (keyword_subreddit_summary["frequency"] - keyword_subreddit_summary["relevant"]).astype(int)

    available_keywords = keyword_subreddit_summary["keyword"].unique()
    selected_keyword = st.selectbox("Choose a keyword to inspect:", sorted(available_keywords))

    filtered_df = keyword_subreddit_summary[keyword_subreddit_summary["keyword"] == selected_keyword]
    filtered_melted = filtered_df.melt(
        id_vars=["subreddit", "keyword"],
        value_vars=["relevant", "irrelevant"],
        var_name="label",
        value_name="count"
    )

    sub_chart = alt.Chart(filtered_melted).mark_bar().encode(
        x=alt.X('subreddit:N', title='Subreddit', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('count:Q', title='Frequency'),
        color='label:N',
        tooltip=['subreddit', 'label', 'count']
    ).properties(
        width=40 * len(filtered_df),
        height=400
    )

    html(f"""
    <div style="overflow-x:auto; border:1px solid #ccc; padding:1rem;">
        {sub_chart.to_html()}
    </div>
    """, height=500)

from collections import defaultdict
import re
def highlight_and_bold_text(text, keywords, annotation_sources):
    spans = []

    # Collect annotation spans with color
    for k, phrase in annotation_sources.items():
        if not phrase or not isinstance(phrase, str): continue
        field = k.replace("source_", "")
        color = CATEGORY_COLORS.get(field, "#DDDDDD")
        for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end(), 'highlight', color))

    # Collect keyword bold spans (skip if already in annotation)
    keywords_sorted = sorted(set(keywords), key=len, reverse=True)
    for kw in keywords_sorted:
        for match in re.finditer(r"\b" + re.escape(kw) + r"\b", text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end(), 'bold', None))

    # Sort and merge spans (prefer highlight over bold)
    spans.sort(key=lambda x: (x[0], -x[1]))  # sort by start asc, end desc

    # Merge and render
    merged = []
    used = [False] * len(text)
    for start, end, kind, color in spans:
        if any(used[start:end]):
            continue  # overlapping, skip
        for i in range(start, end):
            used[i] = True
        merged.append((start, end, kind, color))

    # Build output string
    result = ""
    last = 0
    for start, end, kind, color in merged:
        result += text[last:start]
        chunk = text[start:end]
        if kind == 'highlight':
            result += f'<span style="background-color:{color}66; padding:1px;">{chunk}</span>'
        elif kind == 'bold':
            result += f'<strong>{chunk}</strong>'
        last = end
    result += text[last:]
    return result

def display_annotation_details(ann_dict):
    for key, color in CATEGORY_COLORS.items():
        if key in ann_dict:
            value = ann_dict[key]
            if isinstance(value, list):
                value = ", ".join(value)
            st.markdown(f"<span style='color:{color}'><strong>{key}:</strong> {value}</span>", unsafe_allow_html=True)

# === Load Data ===
#@st.cache_data
def load_data():
    df = pd.read_csv("Full_Dashboard/data/fulltext_labeled_examples.csv")
    return df

df = load_data()

# === Filter Data ===
filtered_df = df[df["model"] == selected_model]
# if not include_random_kfails:
#     filtered_df = filtered_df[filtered_df["subset"] != "random_kfails"]

if filtered_df.empty:
    st.error("There are no full text examples for this model.")
    st.stop()

# === Category Groups ===
categories = [
    "true_positive", "false_positive", 
    "false_negative", "true_negative", 
    "low_confidence"
]

category_labels = {
    "true_positive": "✅ True Positives",
    "false_positive": "⚠️ False Positives",
    "false_negative": "❌ False Negatives",
    "true_negative": "✔️ True Negatives",
    "low_confidence": "🤔 Low Confidence"
}

def confidence_bin(score):
    try:
        score = float(score)
    except (ValueError, TypeError):
        return score

    if score >= 0.85:
        return "High"
    elif score >= 0.65:
        return "Medium"
    else:
        return "Low"

if page == "📄 Sample Labels":
    # === Category Viewer with Navigation ===
    st.title("Model Output: Text Examples")
    st.markdown("Note: Prediction is the model's decision; Gold is the 'right answer' determined by human coders.")

    for cat in categories:
        cat_df = filtered_df[filtered_df["category"] == cat]

        with st.expander(category_labels[cat] + f" ({len(cat_df)})", expanded=False):
            if cat_df.empty:
                st.warning("No examples available in this category.")
                continue
            # Session state tracker
            key = f"index_{cat}"
            if key not in st.session_state:
                st.session_state[key] = 0

            index = st.session_state[key]
            current_row = cat_df.iloc[index]

            # Navigation Buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Prev", key=f"prev_{cat}"):
                    st.session_state[key] = (index - 1) % len(cat_df)
            with col2:
                if st.button("Next ➡️", key=f"next_{cat}"):
                    st.session_state[key] = (index + 1) % len(cat_df)

          
            # First row: ID, Subreddit, Keyword
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**ID**: `{current_row['id']}`")
            with col2:
                st.markdown(f"**Subreddit**: `{current_row.get('subreddit', 'N/A')}`")
            with col3:
                st.markdown(f"**Keyword**: `{current_row.get('keyword', 'N/A')}`")

            # Second row: Prediction, Gold Label, Confidence
            col4, col5, col6 = st.columns(3)
            with col4:
                st.markdown(f"**Prediction**: `{current_row['predicted_label']}`")
            with col5:
                st.markdown(f"**Gold**: `{current_row['gold_label']}`")
            with col6:
                confidence_value = current_row['confidence']
                confidence_label = confidence_bin(confidence_value)
                st.markdown(f"**Confidence**: `{confidence_label}`")
            # Full Text
            st.markdown(f"**Text:**\n\n{current_row['text']}", unsafe_allow_html=True)



@st.cache_data
def load_annotations(jsonl_path):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            entry["annotation_id"] = f"{entry['id']}_0"
            data.append(entry)
    return data


@st.cache_data
def load_text_map(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["id"], df["text"]))

# --- Tab logic inside Streamlit app ---
if page == "📝 Sample Annotations":
    # Initialize session state variable if not set
    if "nav_index" not in st.session_state:
        st.session_state.nav_index = 0

    st.subheader("📄 Sample Text & Annotations")

    annotations = load_annotations("Full_Dashboard/data/annotated_comments.jsonl")
    annotations = [a for a in annotations if a.get("annotations")]
    text_map = load_text_map("Full_Dashboard/data/annotated_text_lookup.csv")  # CSV with id,text

    if not annotations:
        st.error("No annotations found.")
        st.stop()

    # Navigation control
    if "annotation_index" not in st.session_state:
        st.session_state.annotation_index = 0

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Prev"):
            st.session_state.annotation_index = (st.session_state.annotation_index - 1) % len(annotations)
    with col2:
        if st.button("Next ➡️"):
            st.session_state.annotation_index = (st.session_state.annotation_index + 1) % len(annotations)

    entry = annotations[st.session_state.annotation_index]
    post_id = entry["id"]
    raw_text = text_map.get(post_id, "[Text not found]")

    st.markdown(f"**Entry {st.session_state.annotation_index + 1} of {len(annotations)}** | ID: `{post_id}`")

    # Highlighted full text
    annotation_sources = {k: v for k, v in entry.get("annotations", [{}])[0].items() if k.startswith("source_")} if entry.get("annotations") else {}
    highlighted = highlight_and_bold_text(raw_text, [], annotation_sources)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Full Text**")
        st.markdown(f"<div style='line-height:1.6;'>{highlighted}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Annotation Details**")
        if entry.get("annotations"):
            display_annotation_details(entry["annotations"][0])
        else:
            st.markdown("*Marked not relevant.*")



