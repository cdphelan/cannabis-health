import os
import sqlite3
import json
import pandas as pd
import streamlit as st
import re
from dotenv import load_dotenv
from collections import defaultdict


# Load environment variables
load_dotenv()

# DB connection
conn = sqlite3.connect("../../data_collection/reddit_combined.db")
cur = conn.cursor()

# Color mapping for annotation fields
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

@st.cache_data
def get_text_by_id(post_id):
    cur.execute("SELECT title, selftext FROM posts WHERE id=?", (post_id,))
    post = cur.fetchone()
    if post:
        title, selftext = post
        return f"{title}\n\n{selftext}"
    cur.execute("SELECT body FROM comments WHERE id=?", (post_id,))
    comment = cur.fetchone()
    return comment[0] if comment else ""

@st.cache_data
def load_annotations(jsonl_path):
    data = []
    annotation_counter = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            full_text = get_text_by_id(entry["id"])
            annotations = entry.get("annotations", [])
            if annotations:
                for ann in annotations:
                    data.append({
                        "id": entry["id"],
                        "annotation_id": f"{entry['id']}_{annotation_counter}",
                        "full_text": full_text,
                        **ann
                    })
                    annotation_counter += 1
            else:
                data.append({
                    "id": entry["id"],
                    "annotation_id": f"{entry['id']}_0",
                    "full_text": full_text,
                    "no_annotations": True
                })
    return data

def get_reviewed_ids():
    try:
        reviewed = pd.read_csv("review_results.csv")
        return reviewed["annotation_id"].tolist()
    except (FileNotFoundError, KeyError):
        return []

def highlight_text(text, annotation_sources):
    spans = []
    for k, phrase in annotation_sources.items():
        if not phrase or not isinstance(phrase, str): continue
        field = k.replace("source_", "")
        color = CATEGORY_COLORS.get(field, "#DDDDDD")
        for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end(), color))
    # Merge overlapping spans and sort
    spans.sort()
    merged = []
    for span in spans:
        if not merged or span[0] > merged[-1][1]:
            merged.append(list(span))
        else:
            merged[-1][1] = max(merged[-1][1], span[1])
    # Reconstruct with highlights
    result = ""
    last = 0
    for start, end, color in spans:
        result += text[last:start]
        result += f'<span style="background-color:{color}66; padding:1px;">{text[start:end]}</span>'
        last = end
    result += text[last:]
    return result

def display_annotation_details(annotation):
    st.markdown("**Annotation Details**")
    if annotation.get("no_annotations"):
        st.markdown("*Marked as irrelevant*")
    else:
        for key, color in CATEGORY_COLORS.items():
            if key in annotation and not key.startswith("source_"):
                value = annotation[key]
                if isinstance(value, list):
                    value = ", ".join(value)
                st.markdown(f"<span style='color:{color}'><strong>{key}:</strong> {value}</span>", unsafe_allow_html=True)
        if annotation.get("needs_review"):
            st.markdown("---")
            st.markdown(f"🔍 <strong>Needs Review</strong>: {annotation.get('review_reason', 'no reason provided')}", unsafe_allow_html=True)

# --- MAIN APP ---
st.set_page_config(
    page_title="Annotation Review Interface",
    layout="wide",  # 'centered' or 'wide'
)

annotations = load_annotations('annotated_comments.jsonl')
if not annotations:
    st.warning("No annotations found.")
    st.stop()

reviewed_ids = get_reviewed_ids()
unreviewed = [a for a in annotations if a["annotation_id"] not in reviewed_ids]
if not unreviewed:
    st.success("✅ All annotations have been reviewed.")
    st.stop()

# Track current index
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Title and navigation buttons in one row
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown("### ChatGPT annotation review interface")
with col2:
    if st.button("⬅️ Back"):
        st.session_state.current_index = max(0, st.session_state.current_index - 1)
with col3:
    if st.button("➡️ Next"):
        st.session_state.current_index = min(len(unreviewed) - 1, st.session_state.current_index + 1)

annotation = unreviewed[st.session_state.current_index]
annotation_count = sum(1 for a in annotations if a["id"] == annotation["id"])

st.markdown(f"**Reviewing annotation {st.session_state.current_index + 1} of {len(unreviewed)}** | Text ID: `{annotation['id']}` | Annotations in this text: {annotation_count}")

# Review
st.markdown("**Your Review**")
review_col1, review_col2, review_col3 = st.columns([1, 1, 1])

with review_col1:
    relevant = st.radio("Relevant?", ["Yes", "No"], key="relevant_radio", horizontal=True)

with review_col2:
    correct = st.radio("Annotations correct?", ["Yes", "No"], key="correct_radio", horizontal=True)

with review_col3:
    notes_key = f"gpt_notes_{annotation['id']}"
    notes = st.text_input("📝 Notes on GPT annotation (optional)", key=notes_key)

relevant = int(relevant == "Yes")
correct = int(correct == "Yes")


if st.button("✅ Submit Review"):
    new_row = pd.DataFrame([{
        "annotation_id": annotation["annotation_id"],
        "id": annotation["id"],
        "relevant": relevant,
        "annotations_correct": correct,
        "gpt_notes": st.session_state.get(notes_key, "")

    }])
    if os.path.exists("review_results.csv"):
        existing = pd.read_csv("review_results.csv")
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row
    updated.to_csv("review_results.csv", index=False)
    st.success("Review saved.")

    # Advance to next
    st.session_state.current_index += 1
    st.rerun()

col1, col2 = st.columns([2, 1])  # Wider text panel, narrower details panel

with col1:
    st.markdown("**Full Text**")
    highlighted = highlight_text(
    annotation["full_text"],
        {
            k: v for k, v in annotation.items()
            if k.startswith("source_") and isinstance(v, str) and v.strip()
        }
    )
    st.markdown(f"<div style='line-height:1.6;'>{highlighted}</div>", unsafe_allow_html=True)

with col2:
    display_annotation_details(annotation)


