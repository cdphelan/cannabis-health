import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load shared data
@st.cache_data
def load_data():
    df = pd.read_csv("reddit_lines_with_gpt_relevance.csv")  # Assumes columns: id, subreddit, keyword, label
    df["keyword"] = df["keyword"].fillna("")
    df = df.assign(keyword=df["keyword"].str.split(";")).explode("keyword")
    df["keyword"] = df["keyword"].str.strip().str.lower()
    return df

df_raw = load_data()

# Sidebar page navigation
page = st.sidebar.radio("Select a Page", ["📈 Diagnostics", "📊 Keyword Frequency"])

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
        columns={"count": "volume", "mean": "hit_rate"})
    subreddit_summary["usefulness_score"] = subreddit_summary["hit_rate"] * np.log(subreddit_summary["volume"] + 1)

    filtered_subs = subreddit_summary[
        (subreddit_summary["volume"] >= min_sub_freq) &
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
    total = len(df_filtered)
    relevant = df_filtered["relevance"].sum()
    total_relevant_all = df_raw["relevance"].sum()
    precision = relevant / total if total else 0
    recall = relevant / total_relevant_all if total_relevant_all else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🔍 Total Filtered", total)
    col2.metric("✅ Relevant Captured", int(relevant))
    col3.metric("🎯 Precision", f"{precision:.2%}")
    col4.metric("📈 Recall", f"{recall:.2%}")
    col5.metric("📊 F1 Score", f"{f1:.2f}")

    # Tables
    st.subheader("🧩 Keyword Usefulness Table")
    st.dataframe(keyword_summary.loc[filtered_keywords].sort_values("usefulness_score", ascending=False).reset_index(),
                use_container_width=True)

    st.subheader("📌 Subreddit Usefulness Table")
    st.dataframe(subreddit_summary.loc[filtered_subs].sort_values("usefulness_score", ascending=False).reset_index(),
                use_container_width=True)

    # Scatterplots
    col_kws, col_subs = st.columns(2)
    sorted_keywords = keyword_summary.loc[filtered_keywords].sort_values("usefulness_score", ascending=False)
    sorted_keywords["keyword"] = pd.Categorical(sorted_keywords.index, categories=sorted_keywords.index, ordered=True)
    with col_kws:
        st.markdown("### 🔹 Keyword Usefulness Scatter")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=sorted_keywords, x="keyword", y="usefulness_score", ax=ax)
        ax.set_title("Keyword Usefulness")
        ax.set_xlabel("Keyword")
        ax.set_ylabel("Usefulness Score")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    sorted_subs = subreddit_summary.loc[filtered_subs].sort_values("usefulness_score", ascending=False)
    sorted_subs["subreddit"] = pd.Categorical(sorted_subs.index, categories=sorted_subs.index, ordered=True)
    with col_subs:
        st.markdown("### 🔸 Subreddit Usefulness Scatter")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=sorted_subs, x="subreddit", y="usefulness_score", ax=ax2)
        ax2.set_title("Subreddit Usefulness")
        ax2.set_xlabel("Subreddit")
        ax2.set_ylabel("Usefulness Score")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

elif page == "📊 Keyword Frequency":
    st.header("Keyword Frequency")

    # Load data
    #@st.cache_data
    def load_data():
        freq_by_subreddit = pd.read_csv("freq_by_subreddit.csv", index_col=0)
        freq_by_keyword = pd.read_csv("freq_by_keyword.csv", index_col=0)
        keyword_frequencies = pd.read_csv("keyword_frequencies.csv", index_col=0)
        keyword_hit_rates = pd.read_csv("keyword_hit_rates.csv")
        subreddit_hit_rates = pd.read_csv("subreddit_hit_rates.csv", index_col=0)
        return freq_by_subreddit, freq_by_keyword, keyword_frequencies, keyword_hit_rates, subreddit_hit_rates

    freq_by_subreddit, freq_by_keyword, keyword_frequencies, keyword_hit_rates, subreddit_hit_rates = load_data()


    # Sidebar filters
    st.sidebar.header("Filters")

    # User-defined minimum frequency threshold
    keyword_totals = keyword_frequencies.sum(axis=0)
    min_freq = st.sidebar.number_input("Minimum keyword frequency", min_value=1, max_value=10000, value=20, step=1)
    frequent_keywords = keyword_totals[keyword_totals >= min_freq].index.tolist()

    all_keywords = frequent_keywords
    all_subreddits = list(freq_by_subreddit.index)

    selected_subreddits = st.sidebar.multiselect("Subreddits", all_subreddits, default=all_subreddits)

    # Filtered data
    filtered_freq_sub = freq_by_subreddit.loc[selected_subreddits, all_keywords]
    filtered_freq_kw = freq_by_keyword.loc[selected_subreddits, all_keywords]

    # Frequency table
    keyword_total_table = keyword_totals.reset_index()
    keyword_total_table.columns = ["Keyword", "Total Count"]
    keyword_total_table = keyword_total_table[keyword_total_table["Keyword"].isin(all_keywords)]
    keyword_total_table = keyword_total_table[keyword_total_table["Total Count"] >= min_freq]
    keyword_total_table = keyword_total_table.sort_values("Total Count", ascending=False)

    # Keyword usefulness score
    keyword_hit_rates["usefulness_score"] = keyword_hit_rates.apply(
        lambda row: row["hit_rate"] * np.log(row["count"] + 1) if pd.notnull(row["hit_rate"]) and pd.notnull(row["count"]) else 0,
        axis=1
    )
    keyword_usefulness_table = keyword_hit_rates[["keyword", "count", "hit_rate", "usefulness_score"]].sort_values("usefulness_score", ascending=False)

    # Subreddit usefulness score
    subreddit_hit_rates["usefulness_score"] = subreddit_hit_rates.apply(
        lambda row: row["hit_rate"] * np.log(row["count"] + 1) if pd.notnull(row["hit_rate"]) and pd.notnull(row["count"]) else 0,
        axis=1
    )
    subreddit_usefulness_table = subreddit_hit_rates[["hit_rate", "count", "usefulness_score"]].sort_values("usefulness_score", ascending=False)
    subreddit_usefulness_table.index.name = "subreddit"
    subreddit_usefulness_table.reset_index(inplace=True)

    # Plotting helper
    def plot_heatmap(data, title, cmap):
        if data.empty:
            st.warning("No data to display. Please select at least one subreddit and one keyword.")
            return
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(data.T, cmap=cmap, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Overall frequencies",
        "🔠 Frequency by Subreddit",
        "🔡 Frequency by Keyword"
        
    ])

    with tab1:
        if keyword_total_table.empty:
            st.warning("No keywords meet the selected frequency threshold.")
        else:
            st.dataframe(keyword_total_table.reset_index(drop=True), use_container_width=True)

    with tab2:
        with st.expander("ℹ️ How to read this heatmap"):
            st.markdown("""
            - The **darkest cell in a column** shows the subreddit where that keyword is used most frequently.
            - The **darkest cell in a row** shows the keyword used most frequently in that subreddit.
            - The **darkest cell in the table overall** highlights the keyword–subreddit pair with the highest normalized frequency.
            """)
        plot_heatmap(filtered_freq_sub, "Normalized Keyword Frequency by Subreddit", cmap="YlGnBu")


    with tab3:
        with st.expander("ℹ️ How to read this heatmap"):
            st.markdown("""
            - The **darkest cell in a column** shows the subreddit where that keyword appears most frequently across the sample.
            - The **darkest cell in a row** shows the keyword that dominates that subreddit compared to others.
            - The **darkest cell in the table overall** shows the strongest keyword–subreddit connection.
            """)
        plot_heatmap(filtered_freq_kw, "Normalized Subreddit Frequency by Keyword", cmap="YlGnBu")


