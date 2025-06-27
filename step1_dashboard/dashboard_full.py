import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import requests
import altair as alt
import re


# Hidden file ID (you could also store this in st.secrets)
# FILE_ID = st.secrets["gdrive"]["file_id"]

st.set_page_config(layout="wide")

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

# Load shared data
# TODO(dev): Handle missing relevance values more explicitly before production
# TODO(dev): better handling of the keyword_matches table, which currently needs to be manually refreshed with populate_keyword_matches.py
# @st.cache_data
def load_data():
    try: #try checking for local data first
        df = pd.read_csv("../data/all_cannabis-health_with_predictions.csv")  # Assumes columns: id, subreddit, keyword, label
    except:
        # download (for deployed version)
        download_url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        response = requests.get(download_url)
        if response.status_code != 200:
            st.error(f"Failed to download data: {response.status_code}")
            return pd.DataFrame()

        # read into df
        csv_content = io.StringIO(response.text)
        df = pd.read_csv(csv_content)

    # wrangling
    df["keyword"] = df["keyword"].fillna("")
    df = df.assign(keyword=df["keyword"].str.split(";")).explode("keyword")
    df["keyword"] = df["keyword"].str.strip().str.lower()

    return df

df_raw = load_data()
gold_df = pd.read_csv("../data/silver_standard_codes.csv")
# df_eval = df_raw.merge(gold_df, on="id", suffixes=("_pred", "_true"))


# Sidebar page navigation
page = st.sidebar.radio("Select a Page", ["📈 Diagnostics", "📄 Sample Labels", "📄 Sample Annotations"])

#select which model to use
models = df_raw["model"].unique()
selected_model = st.sidebar.selectbox("Select learning model:", sorted(models))

#toggle the keyword filter fails on and off
include_random_kfails = st.sidebar.checkbox("Include random_kfails (non-keyword matches)", value=True)
with st.sidebar.expander("ℹ️ What is random_kfails?"):
    st.markdown(
        "These are random examples that failed the keyword filter. "
        "Toggling this allows you to see how much the filter improves relevance."
    )

#hand over data (filtered on model + keyword fails)
df_raw = df_raw[df_raw["model"] == selected_model]
if not include_random_kfails:
    df_raw = df_raw[df_raw["subset"] != "random_kfails"]


if page == "📈 Diagnostics":
    st.header("🧪 Diagnostic View")

    with st.expander("ℹ️ How to use this page"):
        st.markdown("""
        Use the filter sliders on the left to experiment with different inclusion thresholds.
        - **Precision** measures accuracy. What proportion of the data is relevant?
        - **Recall** measures completeness. What proportion of the relevant data is captured?
        - **F1 score** combines the above.
        
        The general goal is to maximize the F1 score. The subjective judgment call is to decide what balance of signal to noise is best for our needs.
        """)
    # Filters
    st.sidebar.subheader("🔧 Threshold Filters")

    min_kw_freq = st.sidebar.number_input("Min Keyword Frequency", min_value=1, value=5)
    min_kw_hit = st.sidebar.slider("Min Keyword Hit Rate", 0.0, 1.0, 0.0)
    min_kw_score = st.sidebar.slider("Min Keyword Usefulness", 0.0, 10.0, 0.0)

    min_sub_freq = st.sidebar.number_input("Min Subreddit Frequency", min_value=1, value=5)
    min_sub_hit = st.sidebar.slider("Min Subreddit Hit Rate", 0.0, 1.0, 0.0)
    min_sub_score = st.sidebar.slider("Min Subreddit Usefulness", 0.0, 10.0, 0.0)

    # Subreddit diagnostics
    subreddit_summary = df_raw.groupby("subreddit")["relevance"].agg(["count", "mean"]).rename(
        columns={"count": "frequency", "mean": "hit_rate"})
    subreddit_summary["usefulness_score"] = subreddit_summary["hit_rate"] * np.log(subreddit_summary["frequency"] + 1)

    filtered_subs = subreddit_summary[
        (subreddit_summary["frequency"] >= min_sub_freq) &
        (subreddit_summary["hit_rate"] >= min_sub_hit) &
        (subreddit_summary["usefulness_score"] >= min_sub_score)
    ].index.tolist()

    df_sub = df_raw[df_raw["subreddit"].isin(filtered_subs)]

    # Keyword diagnostics after subreddit filtering
    keyword_summary = df_sub.groupby("keyword")["relevance"].agg(["count", "mean"]).rename(
        columns={"count": "frequency", "mean": "hit_rate"})
    keyword_summary["usefulness_score"] = keyword_summary["hit_rate"] * np.log(keyword_summary["frequency"] + 1)

    filtered_keywords = keyword_summary[
        (keyword_summary["frequency"] >= min_kw_freq) &
        (keyword_summary["hit_rate"] >= min_kw_hit) &
        (keyword_summary["usefulness_score"] >= min_kw_score)
    ].index.tolist()

    df_filtered = df_sub[df_sub["keyword"].isin(filtered_keywords)]

    # Diagnostic Metrics
    st.subheader("📈 Diagnostic Metrics")
    # Rename prediction and gold label columns before merge
    df_filtered_2 = df_filtered.rename(columns={"relevance": "relevance_pred"}).copy()
    gold_df = gold_df.rename(columns={"relevant": "relevance_true"})

    # Now merge
    df_filtered_unique = df_filtered_2.drop_duplicates(subset="id")
    df_eval_filtered = df_filtered_unique.merge(gold_df, on="id", how="inner")

    # Diagnostic Metrics
    # total = len(df_eval_filtered)
    true_positives = ((df_eval_filtered["relevance_pred"] == 1) & (df_eval_filtered["relevance_true"] == 1)).sum()
    predicted_positives = (df_eval_filtered["relevance_pred"] == 1).sum()
    actual_positives = (df_eval_filtered["relevance_true"] == 1).sum()

    precision = true_positives / predicted_positives if predicted_positives else 0
    recall = true_positives / actual_positives if actual_positives else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    total = len(df_filtered)
    relevant = df_filtered["relevance"].sum()

    # Row 1: Raw prediction stats
    col_a1, col_a2, col_a3 = st.columns(3)
    col_a1.metric("📦 Filtered Rows", total)
    col_a2.metric("🤖 Predicted Relevant", int(relevant))
    col_a3.metric("🔍 From Keyword Match to Semantic Relevance", f"{int(relevant) / total:.2%}")

    # Row 2: Evaluation stats
    st.markdown("### ✅ Evaluation Against Gold Labels")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🔍 Matched Rows", len(df_eval_filtered))
    col2.metric("✅ True Positives", true_positives)
    col3.metric("🎯 Precision (Gold)", f"{precision:.2%}")
    col4.metric("📈 Recall (Gold)", f"{recall:.2%}")
    col5.metric("📊 F1 Score (Gold)", f"{f1:.2f}")

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
    # Step 1: Filter and sort
    sorted_keywords = keyword_summary.loc[filtered_keywords].sort_values("frequency", ascending=False).copy()

    # Step 2: Compute relevant and irrelevant counts
    sorted_keywords["relevant"] = (sorted_keywords["frequency"] * sorted_keywords["hit_rate"]).round().astype(int)
    sorted_keywords["irrelevant"] = (sorted_keywords["frequency"] - sorted_keywords["relevant"]).astype(int)

    # Step 3: Reset index to expose keyword as a column
    sorted_keywords = sorted_keywords.reset_index()

    # Step 4: Melt for grouped bar chart
    keyword_freq_melted = sorted_keywords.melt(
        id_vars="keyword", 
        value_vars=["relevant", "irrelevant"], 
        var_name="label", 
        value_name="count"
    )

    from streamlit.components.v1 import html

    keyword_chart = alt.Chart(keyword_freq_melted).mark_bar().encode(
        x=alt.X('keyword:N', title='Keyword'),
        y=alt.Y('count:Q', title='Frequency'),
        color='label:N',
        tooltip=['keyword', 'label', 'count']
    ).properties(width=20 * len(sorted_keywords), height=400)  # auto-expand chart width by keyword count

    # Export chart to HTML
    chart_html = keyword_chart.to_html()

    # Display with scroll
    html(f"""
    <div style="overflow-x:auto; border:1px solid #ccc; padding:1rem;">
        {chart_html}
    </div>
    """, height=500)

    # === Grouped Bar Chart: Subreddit Distribution for Selected Keyword ===
    st.markdown("### 🔍 Subreddit Distribution for Selected Keyword")

    # Step 1: Compute (keyword, subreddit) relevance breakdown
    keyword_subreddit_summary = (
        df_filtered
        .groupby(["keyword", "subreddit"])["relevance"]
        .agg(["count", "mean"])
        .rename(columns={"count": "frequency", "mean": "hit_rate"})
        .reset_index()
    )
    keyword_subreddit_summary["relevant"] = (keyword_subreddit_summary["frequency"] * keyword_subreddit_summary["hit_rate"]).round().astype(int)
    keyword_subreddit_summary["irrelevant"] = (keyword_subreddit_summary["frequency"] - keyword_subreddit_summary["relevant"]).astype(int)

    # Step 2: Select a keyword from dropdown
    available_keywords = keyword_subreddit_summary["keyword"].unique()
    selected_keyword = st.selectbox("Choose a keyword to inspect:", sorted(available_keywords))

    # Step 3: Melt for plotting
    filtered_df = keyword_subreddit_summary[keyword_subreddit_summary["keyword"] == selected_keyword]
    filtered_melted = filtered_df.melt(
        id_vars=["subreddit", "keyword"],
        value_vars=["relevant", "irrelevant"],
        var_name="label",
        value_name="count"
    )

    # Step 4: Render chart with horizontal scroll support
    sub_chart = alt.Chart(filtered_melted).mark_bar().encode(
        x=alt.X('subreddit:N', title='Subreddit', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('count:Q', title='Frequency'),
        color='label:N',
        tooltip=['subreddit', 'label', 'count']
    ).properties(
        width=40 * len(filtered_df),  # scale width to number of subreddits
        height=400
    )

    html(f"""
    <div style="overflow-x:auto; border:1px solid #ccc; padding:1rem;">
        {sub_chart.to_html()}
    </div>
    """, height=500)


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
    st.markdown("**Annotation Details**")
    for key, color in CATEGORY_COLORS.items():
        if key in ann_dict:
            value = ann_dict[key]
            if isinstance(value, list):
                value = ", ".join(value)
            st.markdown(f"<span style='color:{color}'><strong>{key}:</strong> {value}</span>", unsafe_allow_html=True)

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("../../../data_collection/fulltext_examples.csv")
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
    "low_confidence", "controversial"
]

category_labels = {
    "true_positive": "✅ True Positives",
    "false_positive": "⚠️ False Positives",
    "false_negative": "❌ False Negatives",
    "true_negative": "✔️ True Negatives",
    "low_confidence": "🤔 Low Confidence",
    "controversial": "🔥 Controversial"
}

def confidence_bin(score):
    try:
        score = float(score)
    except (ValueError, TypeError):
        return score

    if score >= 0.85:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"

if page == "📄 Sample Labels":
    # === Category Viewer with Navigation ===
    st.title("🧾 Model Output: Text Examples")

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
    annotations_df = pd.read_json(jsonl_path, lines=True)
    return annotations_df

# --- Tab logic inside Streamlit app ---
if page == "📄 Sample Annotations":
    annotation_path = "data/annotated_comments.jsonl"  # update as needed
    filtered_df = load_annotations(annotation_path)

    # Navigation buttons
    st.subheader("📄 Sample Text & Annotations")
    col_nav1, col_nav2, col_jump = st.columns([1, 2, 4])

    with col_nav1:
        if st.button("⬅️ Back"):
            st.session_state.nav_index = max(0, st.session_state.nav_index - 1)
    with col_nav2:
        if st.button("➡️ Next"):
            st.session_state.nav_index = min(len(filtered_df) - 1, st.session_state.nav_index + 1)
    with col_jump:
        post_ids = filtered_df["id"].tolist()
        selected_id = st.selectbox(
            "Jump to Post ID", 
            post_ids, 
            index=st.session_state.nav_index, 
            label_visibility="collapsed"  # Hide label to fix vertical alignment
        )
        st.session_state.nav_index = post_ids.index(selected_id)

    # Validate navigation index
    st.session_state.nav_index = min(st.session_state.nav_index, len(filtered_df) - 1)
    # Get entry row
    row = filtered_df[filtered_df["id"] == selected_id].iloc[0]
    text = row["text"]
    prediction = "Relevant" if row["relevance"] else "Not Relevant"
    ground_truth = "Relevant" if row.get("gold_label") == 1 else ("Not Relevant" if row.get("gold_label") == 0 else "--")
    confidence = row.get("confidence", None)

    # Classification panel
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**Subreddit:** {row.get('subreddit', '')}")
        
    with col2:
        st.markdown(f"**Keyword(s):** {row.get('keyword', '')}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Prediction", prediction)
    col2.metric("Ground Truth", ground_truth)
    if confidence:
        col3.metric("Model Confidence", f"{confidence:.2%}")
    else:
        col3.metric("Model Confidence", "--")

    # Text display with highlighting
    col1, col2 = st.columns([2, 1])
    with col1:
        # Prepare inputs
        text = row.get("text", "")
        keywords_raw = row.get("keyword", "")
        keywords = [kw.strip() for kw in keywords_raw.split(";") if kw.strip()]
        annotations = row.get("annotations", {})
        source_phrases = {k: v for k, v in annotations.items() if k.startswith("source_")}

        # Apply combined highlighting
        final_text = highlight_and_bold_text(text, keywords, source_phrases)

        # Render
        st.markdown("**Full Text**")
        st.markdown(f"<div style='line-height:1.6;'>{final_text}</div>", unsafe_allow_html=True)

    with col2:
        if annotations:
            display_annotation_details(annotations)
        else:
            st.markdown("*No annotation data available.*")



