
import streamlit as st
import pandas as pd
import re
import streamlit.components.v1 as components

st.set_page_config(page_title="Cannabis Keyword Labeling", layout="centered")

# Inject keyboard shortcut script using exact button matches
components.html("""
<script>
console.log("✅ Keyboard script attached (exact match)");

window.parent.document.addEventListener('keydown', function(event) {
  console.log("Key pressed from parent:", event.key);

  const buttons = [...window.parent.document.querySelectorAll('button')];

  if (event.key === 'a') {
      buttons.find(el => el.innerText.trim().toLowerCase() === '✅ relevant (a)')?.click();
  } else if (event.key === 's') {
      buttons.find(el => el.innerText.trim().toLowerCase() === '❌ irrelevant (s)')?.click();
  } else if (event.key === 'ArrowLeft') {
      buttons.find(el => el.innerText.trim().toLowerCase() === '⬅️ back')?.click();
  }
});
</script>
""", height=0)

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
    unlabeled = st.session_state.df['label'].isna()
    first_unlabeled = unlabeled.idxmax() if unlabeled.any() else 0
    st.session_state.index = first_unlabeled

def highlight_keywords(text, keywords):
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f"<mark>{kw}</mark>", text)
    return text

def save_progress():
    st.session_state.df.to_csv("reddit_labeled_output.csv", index=False)

# Display current item
df = st.session_state.df
if st.session_state.index < len(df):
    row = df.iloc[st.session_state.index]
    st.markdown(f"**Subreddit:** {row['subreddit']}")
    keywords = [kw.strip() for kw in row['keyword'].split(';')] if pd.notna(row['keyword']) else []
    combined_text = "\n\n".join([
        str(row.get('title', '')),
        str(row.get('post_text', '')),
        str(row.get('comment_body', '')),
        str(row.get('reply_body', '')) 
    ]).strip()

    # highlighted = highlight_keywords(row['text'], keywords) #for 1 column (comment out combined_text)
    highlighted = highlight_keywords(combined_text, keywords) #multiple input columns
    st.markdown(highlighted, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    if col1.button("⬅️ Back", key="back"):
        if st.session_state.index > 0:
            st.session_state.index -= 1
    if col2.button("❌ Irrelevant (A)", key="irrelevant"):
        df.at[st.session_state.index, 'label'] = 0
        st.session_state.index += 1
        save_progress()
    if col3.button("✅ Relevant (S)", key="relevant"):
        df.at[st.session_state.index, 'label'] = 1
        st.session_state.index += 1
        save_progress()
else:
    st.success("🎉 All entries labeled!")

if st.button("💾 Export to CSV"):
    save_progress()
    st.success("Saved to reddit_labeled_output.csv")
