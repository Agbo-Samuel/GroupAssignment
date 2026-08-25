import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
import ssl
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pyarrow import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, auc
from PIL import Image


################# Global Setups ###################################
st.set_page_config(page_title="Multilingual Sentiment Analysis", page_icon="📈", layout="wide")
#

st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <h1>🌍 Multilingual Sentiment Analysis</h1>
        <p style="color:gray;">FastText embeddings · SVM & Logistic Regression · Cross-lingual insights</p>
    </div>
""", unsafe_allow_html=True)

#




## SSL context for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context

## Download required NLTK packages
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


## Load + parse dataset (cached so it only runs once, not on every rerun)
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    dataset = base_dir / "multilingual_sentiment_train.csv"

    with open(dataset, "r", encoding="UTF-8") as file:
        lines = file.readlines()

    parsed_data = []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            #FIRST = TEXT
            #SECOND = CLASS
            text, class_label = parts
            parsed_data.append([text, class_label])
        else:
            parsed_data.append([line.strip(), "Unknown"])

    df = pd.DataFrame(parsed_data, columns=["text", "label"])
    return lines, df

# ---------------------------------------
# Data cleaning (cached, returns a NEW dataframe instead of mutating a shared one)
@st.cache_data
def clean_text(df):
    cleaned = []
    for text in df["text"]:
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)      # Remove HTML
        text = re.sub(r'[^\w\s]', '', text)    # Remove punctuation
        text = re.sub(r'\d+', '', text)        # Remove digits
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        cleaned.append(text)

    df = df.copy()
    df["Cleaned"] = cleaned
    return df


# Load once at the top — this is just data prep, no st.write/st.dataframe here
lines, df = load_data()
df_clean = clean_text(df)


# ---------------------------------------
# PAGE 1 - RAW DATA
def page1():
    st.subheader("Raw Unprocessed Data")
    st.write(lines)


# PAGE 2 - DATA CLEANING
def page2():
    st.subheader("Data Cleaning")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df_clean))
    #col2.metric("Languages Detected", df_clean["lang"].nunique())
    #col3.metric("Avg. Text Length", f"{df_clean['text'].str.len().mean():.0f} chars")


    st.dataframe(df_clean)


def page3():
    st.subheader("Exploratory Text Analysis")


def page4():
    st.subheader("Text representation")


# SIDEBAR NAVIGATION
pages = {
    "Raw Unprocessed Data": page1,
    "Data Cleaning": page2,
    "Exploratory Text Analysis": page3,
    "Text representation": page4,
}

#selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
#pages[selected_page]()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Raw Data", "🧹 Cleaning", "🔍 EDA", "🔤 Representation"])
with tab1:
    page1()
with tab2:
    page2()