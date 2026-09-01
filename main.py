import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
import ssl
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, auc
from PIL import Image
import plotly.express as px


################# Global Setups ###################################
st.set_page_config(page_title="Multilingual Sentiment Analysis", page_icon="📈", layout="wide")

st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <h1>🌍 Multilingual Sentiment Analysis</h1>
        <p style="color:gray;">FastText embeddings · SVM & Logistic Regression · Cross-lingual insights</p>
    </div>
""", unsafe_allow_html=True)

## SSL context for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context

## Download required NLTK packages
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


# ---------------------------------------
# Load dataset (cached so it only runs once, not on every rerun).
#
# NOTE: the source file already has a proper header ("label,text,lang") and
# is comma-delimited, so we can just use pd.read_csv() directly - no need to
# manually split lines on tabs. This also loads the pre-sampled file
# (multilingual_sentiment_sample.csv), which was stratified by language and
# label ahead of time so every language is represented without pulling in
# the full ~226k-row source file. See stratified_sample.py for how that
# sample was generated - rerun it with a different PER_LANG_CAP if you want
# a bigger or smaller dataset.
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    dataset = base_dir / "multilingual_sentiment_sample.csv"
    df = pd.read_csv(dataset)
    return df


# ---------------------------------------
# Data cleaning (cached, returns a NEW dataframe instead of mutating a shared one)
@st.cache_data
def clean_text(df):
    cleaned = []
    for text in df["text"]:
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)         # Remove HTML
        text = re.sub(r'[^\w\s]', '', text)       # Remove punctuation
        text = re.sub(r'\d+', '', text)           # Remove digits
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        cleaned.append(text)

    df = df.copy()
    df["Cleaned"] = cleaned
    return df


df = load_data()
df_clean = clean_text(df)


@st.cache_data
def dedupe_data(df):
    shape_before = df.shape
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    shape_after = df_dedup.shape
    return df_dedup, shape_before, shape_after


# ---------------------------------------
# PAGE 1 - RAW DATA
def page1():
    st.subheader("Raw Unprocessed Data")
    st.caption(f"{len(df):,} rows · {df['lang'].nunique()} languages (pre-sampled for app performance).")

    # Never dump the entire dataset to the page with st.write() - rendering
    # every row into the DOM at once is what was crashing the app on the
    # full file. Show a bounded preview instead.
    preview_rows = st.slider("Rows to preview", 10, min(500, len(df)), min(50, len(df)))
    st.dataframe(df.head(preview_rows), use_container_width=True)


# PAGE 2 - DATA CLEANING
def page2():
    st.subheader("Data Cleaning")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df_clean))
    col2.metric("Languages", df_clean["lang"].nunique())
    col3.metric("Avg. Text Length", f"{df_clean['text'].astype(str).str.len().mean():.0f} chars")

    st.dataframe(df_clean.head(200), use_container_width=True)


def page3():
    st.subheader("Exploratory Text Analysis")

    lang_counts = df_clean["lang"].value_counts().reset_index()
    lang_counts.columns = ["lang", "count"]
    fig = px.bar(lang_counts, x="lang", y="count", title="Rows per Language")
    st.plotly_chart(fig, use_container_width=True)

    label_counts = df_clean["label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]
    fig2 = px.pie(label_counts, names="label", values="count", title="Sentiment Label Balance")
    st.plotly_chart(fig2, use_container_width=True)


def page4():
    st.subheader("Text representation")


# SIDEBAR NAVIGATION
pages = {
    "Raw Unprocessed Data": page1,
    "Data Cleaning": page2,
    "Exploratory Text Analysis": page3,
    "Text representation": page4,
}

tab1, tab2, tab3, tab4 = st.tabs(["📊 Raw Data", "🧹 Cleaning", "🔍 EDA", "🔤 Representation"])
with tab1:
    page1()
with tab2:
    page2()
with tab3:
    page3()
with tab4:
    page4()
