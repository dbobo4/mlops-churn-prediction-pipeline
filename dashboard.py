# dashboard.py

import os
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# import the three generated report paths
from report_generator import (
    REPORT_PATH,
    TESTS_PATH,
    SUITE_PATH,
    TRAIN_PREVIEW_PATH,
    EVALUATE_PREVIEW_PATH,
)

# Basic page config
st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")
st.title("📊 Model Monitoring Dashboard")

st.markdown(
    """
    This dashboard lets you browse the **four** key views:

    1. **Inspect**: Preview the training and evaluation data  
    2. **Data Drift**: Evidently drift report  
    3. **Column/Row Quality & Structure Tests**  
    4. **Stability & No-Target Performance Suite**
    """
)

# Create four tabs
tabs = st.tabs(["🔍 Inspect", "📉 Drift Report", "🧪 Quality Tests", "📋 Suite Report"])

# 🔍 Inspect Tab
with tabs[0]:
    st.header("Inspect Input Datasets")
    col1, col2 = st.columns(2)

    # Left: Train data preview
    with col1:
        st.subheader("Train Dataset (first 20 cols & rows)")
        if os.path.exists(TRAIN_PREVIEW_PATH):
            # load the JSON preview produced by report_generator
            with open(TRAIN_PREVIEW_PATH, "r", encoding="utf-8") as f:
                train_records = json.load(f)
            df_train = pd.DataFrame.from_records(train_records)
            st.dataframe(df_train.iloc[:, :20], use_container_width=True)
        else:
            st.warning("No train preview found. Run the API’s `/model/evaluate_batch` endpoint first.")

    # Right: Evaluate data preview
    with col2:
        st.subheader("Evaluate Dataset (first 20 cols & rows)")
        if os.path.exists(EVALUATE_PREVIEW_PATH):
            # load the JSON preview produced by report_generator
            with open(EVALUATE_PREVIEW_PATH, "r", encoding="utf-8") as f:
                eval_records = json.load(f)
            df_eval = pd.DataFrame.from_records(eval_records)
            st.dataframe(df_eval.iloc[:, :20], use_container_width=True)
        else:
            st.warning("No evaluate preview found. Run the API’s `/model/evaluate_batch` endpoint first.")

# 📉 Drift Report Tab
with tabs[1]:
    st.header("Data Drift Report")
    if not os.path.exists(REPORT_PATH):
        st.warning("No drift report found. Run `/model/evaluate_batch` first.")
    else:
        html = open(REPORT_PATH, "r", encoding="utf-8").read()
        components.html(html, height=1000, scrolling=True)

# 🧪 Quality Tests Tab
with tabs[2]:
    st.header("Column/Row Quality & Structure Tests")
    if not os.path.exists(TESTS_PATH):
        st.warning("No tests report found. Run `/model/evaluate_batch` first.")
    else:
        html = open(TESTS_PATH, "r", encoding="utf-8").read()
        components.html(html, height=1000, scrolling=True)

# 📋 Suite Report Tab
with tabs[3]:
    st.header("Stability & No-Target Performance Suite")
    if not os.path.exists(SUITE_PATH):
        st.warning("No suite report found. Run `/model/evaluate_batch` first.")
    else:
        html = open(SUITE_PATH, "r", encoding="utf-8").read()
        components.html(html, height=1000, scrolling=True)
