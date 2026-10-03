import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Public Health EDA", layout="wide")

def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)

def infer_column_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

st.title("📊 Public Health Exploratory Data Analysis (EDA)")

st.write(
    "Upload a public health CSV, then explore missing values, distributions, and relationships "
    "using Pandas + Matplotlib + Seaborn."
)

uploaded_file = st.file_uploader("Upload your public health CSV", type=["csv"])
df = load_data(uploaded_file)

# Fallback local file (optional)
if df is None:
    try:
        df = pd.read_csv("data/public_health.csv")
        st.info("Loaded local file: data/public_health.csv")
    except Exception:
        st.warning("Please upload a CSV file (or add data/public_health.csv).")
        st.stop()

st.divider()

# ---------- Dataset Overview ----------
st.subheader("Dataset Overview")
st.write("Shape:", df.shape)
st.write("Columns:", list(df.columns))

numeric_cols, categorical_cols = infer_column_types(df)

with st.expander("Preview data (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

# ---------- Missing Values ----------
st.subheader("Missing Values")

with st.expander("Missing values heat check"):
    missing = df.isna().sum().sort_values(ascending=False)
    st.write(missing.head(20))

    top_n = min(20, len(missing))
    if top_n > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            x=missing.head(top_n).values,
            y=missing.head(top_n).index,
            ax=ax,
            palette="viridis"
        )
        ax.set_title("Top Missing Values (Top Columns)")
        ax.set_xlabel("Missing Count")
        ax.set_ylabel("Column")
        st.pyplot(fig, clear_figure=True)

# ---------- Optional preprocessing for plots ----------
st.divider()
st.subheader("Exploratory Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Missing Value Treatment (optional)")
    strategy = st.selectbox(
        "How to handle missing values for plotting",
        ["None", "Fill numeric with median", "Fill categorical with mode"],
        index=0
    )

    df_plot = df.copy()
    if strategy == "Fill numeric with median":
        for c in numeric_cols:
            if c in df_plot.columns:
                df_plot[c] = df_plot[c].fillna(df_plot[c].median())
    elif strategy == "Fill categorical with mode":
        for c in categorical_cols:
            mode_series = df_plot[c].mode(dropna=True)
            if len(mode_series) > 0:
                df_plot[c] = df_plot[c].fillna(mode_series.iloc[0])

with col2:
    st.markdown("### Correlation Heatmap (numeric only)")
    corr_method = st.selectbox("Correlation type", ["pearson", "spearman", "kendall"], index=0)

    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns to compute correlations.")
    else:
        corr_df = df_plot[numeric_cols].corr(method=corr_method)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_df, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"Correlation Heatmap ({corr_method})")
        st.pyplot(fig, clear_figure=True)

# ---------- Distributions ----------
st.divider()
st.subheader("Distributions (Numeric Columns)")

if len(numeric_cols) == 0:
    st.warning("No numeric columns found.")
else:
    dist_col = st.selectbox("Pick a numeric column for distribution", numeric_cols)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df_plot[dist_col].dropna(), kde=True, ax=ax, color="#2E86AB")
    ax.set_title(f"Distribution of {dist_col}")
    ax.set_xlabel(dist_col)
    st.pyplot(fig, clear_figure=True)

# ---------- Categorical vs Numeric ----------
st.divider()
st.subheader("Relationships (Categorical vs Numeric)")

if len(categorical_cols) == 0 or len(numeric_cols) == 0:
    st.warning("Need at least one categorical and one numeric column.")
else:
    group_col = st.selectbox("Pick a categorical column (grouping)", categorical_cols)
    target_col = st.selectbox("Pick a numeric column (target)", numeric_cols)

    top_k = st.slider("Top categories to display", min_value=5, max_value=25, value=10)
    top_values = df_plot[group_col].value_counts(dropna=True).head(top_k).index
    tmp = df_plot[df_plot[group_col].isin(top_values)]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=tmp, x=group_col, y=target_col, ax=ax, palette="Set2")
    ax.set_title(f"{target_col} by {group_col} (Top {top_k} categories)")
    ax.set_xlabel(group_col)
    ax.set_ylabel(target_col)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig, clear_figure=True)

# ---------- Summary stats ----------
st.divider()
st.subheader("Summary Statistics")

with st.expander("Numeric summary"):
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
    else:
        st.write("No numeric columns.")

with st.expander("Categorical summary"):
    if len(categorical_cols) > 0:
        for c in categorical_cols:
            st.write(f"Top values for: {c}")
            st.write(df[c].value_counts(dropna=False).head(10))
    else:
        st.write("No categorical columns.")

st.divider()
st.subheader("Next Steps")
st.markdown(
    """
- Add outlier detection (IQR / Z-score)
- Add pairplot / scatter matrix (for selected features)
- Add automatic “target column” discovery (if you define one)
- Add model-ready preprocessing + ML
"""
)
