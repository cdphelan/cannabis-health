#to run:
# streamlit run label_interface.py 

import streamlit as st
import pandas as pd
import re
import streamlit.components.v1 as components

def display_scrollable_text(highlighted_html):
    st.markdown(f"""
    <div style="
        height: 300px;
        overflow-y: auto;
        padding: 0.5em;
        border: 1px solid #ccc;
        background-color: #fdfdfd;
        font-size: 0.95em;
    ">
        {highlighted_html}
    </div>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Cannabis Keyword Labeling", layout="centered")

# Inject keyboard shortcut script using exact button matches
#I just can't get this to work reliably and it's not really that critical
# components.html("""
# <script>
# console.log("✅ Keyboard script attached (exact match)");

# window.parent.document.addEventListener('keydown', function(event) {
#   console.log("Key pressed from parent:", event.key);

#   const buttons = [...window.parent.document.querySelectorAll('button')];

#   if (event.key === 'a') {
#       buttons.find(el => el.innerText.trim().toLowerCase() === '✅ relevant (a)')?.click();
#   } else if (event.key === 's') {
#       buttons.find(el => el.innerText.trim().toLowerCase() === '❌ irrelevant (s)')?.click();
#   } else if (event.key === 'ArrowLeft') {
#       buttons.find(el => el.innerText.trim().toLowerCase() === '⬅️ back')?.click();
#   }
# });
# </script>
# """, height=0)

# Load data and handle pre-labeled rows
if 'df' not in st.session_state:
    df = pd.read_csv("reddit_input.csv")
    if 'label' not in df.columns:
        df['label'] = None
    st.session_state.df = df
    # Save already-labeled rows immediately
    df[df['label'].notna()].to_csv("reddit_labeled_output.csv", index=False)

# Skip to first unlabeled row
if 'index' not in st.session_state:
    st.markdown("🔔 **Happy labeling!**")
    unlabeled = st.session_state.df['label'].isna()
    first_unlabeled = unlabeled.idxmax() if unlabeled.any() else 0
    st.session_state.index = first_unlabeled

def highlight_keywords(text, keywords):
    if not text or not keywords:
        return text

    for kw in keywords:
        if not kw:
            continue
        pattern = re.compile(rf'(?<!\w)({re.escape(kw)})(?!\w)', re.IGNORECASE)
        text = pattern.sub(r'<mark>\g<1></mark>', text)

    return text

def save_progress():
    st.session_state.df.to_csv("reddit_labeled_output.csv", index=False)

# Display current item
df = st.session_state.df
if st.session_state.index < len(df):
    row = df.iloc[st.session_state.index]
    st.markdown(f"**Subreddit:** {row['subreddit']}")
    keywords = [kw.strip() for kw in row['keyword'].split(';')] if pd.notna(row['keyword']) else []
    # combined_text = "\n\n".join([
    #     str(row.get('title', '')),
    #     str(row.get('post_text', '')),
    #     str(row.get('comment_body', '')),
    #     str(row.get('reply_body', '')) 
    # ]).strip()

    highlighted = highlight_keywords(row['text'], keywords) #for 1 column (comment out combined_text)
    # highlighted = highlight_keywords(combined_text, keywords) #multiple input columns
    display_scrollable_text(highlighted)
    #st.markdown(highlighted, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    if col1.button("⬅️ Back", key="back"):
        if st.session_state.index > 0:
            st.session_state.index -= 1
            st.rerun()
    if col2.button("❌ Irrelevant", key="irrelevant"):
        df.at[st.session_state.index, 'label'] = 0
        st.session_state.index += 1
        save_progress()
        st.rerun()

    if col3.button("✅ Relevant", key="relevant"):
        df.at[st.session_state.index, 'label'] = 1
        st.session_state.index += 1
        save_progress()
        st.rerun()

else:
    st.success("🎉 All entries labeled!")

# if st.button("💾 Export to CSV"):
#     save_progress()
#     st.success("Saved to reddit_labeled_output.csv")

# --- Progress Bar ---
total = len(df)
labeled = df['label'].notna().sum()
st.progress(labeled / total)
st.markdown(f"Progress automatically saved. **Labeled {labeled} of {total} entries ({labeled/total:.1%})**")
st.markdown(f"IMPORTANT: close this window in between coding sessions. Some reloading browser behaviors may cause coding progress to be overwritten.")


# --- Keyword Filter ---
# come back to this feature later if you want it