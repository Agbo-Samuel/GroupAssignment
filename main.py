import time
import re
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from gensim.models import FastText
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import plotly.express as px
import plotly.graph_objects as go


################# Global Setups ###################################
st.set_page_config(page_title="Multilingual Sentiment Analysis", page_icon="📈", layout="wide")

st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <h1>🌍 Multilingual Sentiment Analysis</h1>
        <p style="color:gray;"> 📊 Group 15 · FastText embeddings · SVM & Logistic Regression · Cross-lingual insights</p>
        <p style="color:gray;"> 👥 Group Members :  Pascal Adiali · Samuel Agbo · Winifred Korang Takyi · 
        Kweku Afedzi Hayford · Emmanuel Agyapong </p>
    </div>
""", unsafe_allow_html=True)

nltk.download('stopwords', quiet=True)

# nltk's stopword corpus is keyed by full language name and only covers a
# subset of the 29 languages in this dataset. Where a language isn't
# covered, stopword removal is simply skipped for that language's rows -
# tokens are kept as-is. This is a known limitation worth noting in the
# report's interpretation section.
LANG_TO_NLTK = {
    'ind': 'indonesian', 'tur': 'turkish', 'deu': 'german', 'ara': 'arabic',
    'fin': 'finnish', 'cmn': 'chinese', 'zho': 'chinese', 'eng': 'english',
    'rus': 'russian', 'spa': 'spanish', 'nor': 'norwegian', 'ell': 'greek',
    'heb': 'hebrew', 'eus': 'basque',
}


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    dataset = base_dir / "multilingual_sentiment_sample.csv"
    df = pd.read_csv(dataset)
    return df


# =========================================================
# DATA CLEANING (Regex-based)
# =========================================================
@st.cache_data
def clean_text(df):
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)         # Remove HTML
        text = re.sub(r'[^\w\s]', '', text)       # Remove punctuation
        text = re.sub(r'\d+', '', text)           # Remove digits
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    df = df.copy()
    df["Cleaned"] = df["text"].apply(clean)
    return df


# =========================================================
# TEXT PREPROCESSING: Tokenization + stopword removal
# =========================================================
@st.cache_data
def preprocess_text(df):
    df = df.copy()
    # Whitespace tokenization: the source corpus is already word-segmented
    # (including for CJK languages), so a simple split is more reliable
    # across 29 languages than an English-tuned tokenizer like nltk's punkt.
    df["Tokens"] = df["Cleaned"].apply(str.split)

    stopword_cache = {}

    def get_stopwords(lang_code):
        nltk_lang = LANG_TO_NLTK.get(lang_code)
        if nltk_lang is None:
            return frozenset()
        if nltk_lang not in stopword_cache:
            try:
                stopword_cache[nltk_lang] = frozenset(stopwords.words(nltk_lang))
            except Exception:
                stopword_cache[nltk_lang] = frozenset()
        return stopword_cache[nltk_lang]

    tokens_clean = []
    for tokens, lang in zip(df["Tokens"], df["lang"]):
        sw = get_stopwords(lang)
        tokens_clean.append([t for t in tokens if t not in sw] if sw else tokens)
    df["Tokens_clean"] = tokens_clean

    return df


# =========================================================
# TEXT REPRESENTATION: FastText embeddings + document vectors
# =========================================================
@st.cache_resource(show_spinner="Training FastText embeddings (one-time, cached)...")
def train_fasttext(tokens_list, vector_size, epochs, min_count):
    model = FastText(
        sentences=tokens_list,
        vector_size=vector_size,
        window=5,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        sg=1,       # skip-gram
        seed=42,
    )
    return model


def doc_vector(tokens, model, dim):
    vecs = []
    for t in tokens:
        try:
            vecs.append(model.wv[t])  # FastText can embed OOV words via subword n-grams
        except KeyError:
            continue
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


@st.cache_resource(show_spinner="Building document embeddings...")
def build_doc_vectors(_model, tokens_list, vector_size, cache_key):
    # cache_key (vector_size, epochs, min_count) forces recompute when
    # hyperparameters change, since _model itself isn't hashed (leading underscore).
    return np.vstack([doc_vector(t, _model, vector_size) for t in tokens_list])


# =========================================================
# MODELING: SVM + Logistic Regression
# =========================================================
@st.cache_resource(show_spinner="Training SVM and Logistic Regression models...")
def train_models(X_train, y_train, random_state):
    log_reg = LogisticRegression(max_iter=1000, random_state=random_state)
    log_reg.fit(X_train, y_train)

    svm = LinearSVC(random_state=random_state, max_iter=5000)
    svm.fit(X_train, y_train)

    return log_reg, svm


def get_scores(model, X_test):
    """Returns a score usable for ROC-AUC: predict_proba if available, else decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = get_scores(model, X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_score": y_score,
    }


# =========================================================
# LOAD + PROCESS DATA (runs once, cached)
# =========================================================
df = load_data()
df_clean = clean_text(df)
df_prep = preprocess_text(df_clean)


# =========================================================
# PAGE 1 - RAW DATA
# =========================================================
def page1():
    st.subheader("Raw Unprocessed Data")
    st.markdown("""
        This page displays the Data in its raw unprocessed form.
        """)


    st.caption(f"{len(df):,} rows · {df['lang'].nunique()} languages (pre-sampled for app performance).")

    preview_rows = st.slider("Rows to preview", 10, min(500, len(df)), min(50, len(df)))
    st.dataframe(df.head(preview_rows), use_container_width=True)


# =========================================================
# PAGE 2 - DATA CLEANING & PREPROCESSING
# =========================================================
def page2():
    st.subheader("Data Cleaning (Regular Expressions)")
    st.markdown("""
    Cleaning steps applied to raw text: lowercasing, HTML tag removal, punctuation
    removal, digit removal, and whitespace normalization.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df_clean))
    col2.metric("Languages", df_clean["lang"].nunique())
    col3.metric("Avg. Text Length", f"{df_clean['text'].astype(str).str.len().mean():.0f} chars")

    st.dataframe(df_clean[["text", "Cleaned", "label", "lang"]].head(200), use_container_width=True)

    st.subheader("Text Preprocessing (Tokenization + Stopword Removal)")
    n_supported = df_prep["lang"].isin(LANG_TO_NLTK.keys()).sum()
    st.info(
        f"Stopword removal was applied to **{n_supported:,} / {len(df_prep):,}** rows "
        f"(covering {len(LANG_TO_NLTK)} of the {df_prep['lang'].nunique()} languages present, "
        "based on nltk's available stopword corpora). Languages without an nltk stopword list "
        "keep their full token set."
    )
    st.dataframe(
        df_prep[["Cleaned", "Tokens", "Tokens_clean", "lang"]].head(50),
        use_container_width=True
    )


# =========================================================
# PAGE 3 - EXPLORATORY TEXT ANALYSIS
# =========================================================
def page3():
    st.subheader("Exploratory Text Analytics")

    col1, col2 = st.columns(2)
    with col1:
        lang_counts = df_prep["lang"].value_counts().reset_index()
        lang_counts.columns = ["lang", "count"]
        fig = px.bar(lang_counts, x="lang", y="count", title="Rows per Language")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        label_counts = df_prep["label"].value_counts().reset_index()
        label_counts.columns = ["label", "count"]
        fig2 = px.pie(label_counts, names="label", values="count", title="Sentiment Label Balance")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Token Length Distribution")
    df_prep["token_len"] = df_prep["Tokens_clean"].apply(len)
    fig3 = px.histogram(df_prep, x="token_len", nbins=50, title="Tokens per Document (after cleaning)")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Most Frequent Tokens (overall)")
    from collections import Counter
    all_tokens = [t for toks in df_prep["Tokens_clean"] for t in toks]
    top_tokens = Counter(all_tokens).most_common(20)
    top_df = pd.DataFrame(top_tokens, columns=["token", "frequency"])
    fig4 = px.bar(top_df, x="token", y="frequency", title="Top 20 Tokens Across All Languages")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "Note: since this mixes 29 languages, top tokens are dominated by whichever "
        "languages have the most common short function words remaining after cleaning."
    )


# =========================================================
# PAGE 4 - TEXT REPRESENTATION (FastText embeddings)
# =========================================================
FT_VECTOR_SIZE = 64
FT_EPOCHS = 4
FT_MIN_COUNT = 3


def page4():
    st.subheader("Text Representation: FastText Word Embeddings")

    st.markdown("""
    A FastText model is trained directly on this dataset's cleaned, tokenized text.
    FastText represents words using subword (character n-gram) information, which
    helps it handle rare words and multiple languages/scripts better than word2vec.
    Each document's vector is simply the average of its word vectors.
    """)

    tokens_list = df_prep["Tokens_clean"].tolist()

    t0 = time.time()
    ft_model = train_fasttext(tokens_list, FT_VECTOR_SIZE, FT_EPOCHS, FT_MIN_COUNT)
    train_time = time.time() - t0

    cache_key = (FT_VECTOR_SIZE, FT_EPOCHS, FT_MIN_COUNT)
    X = build_doc_vectors(ft_model, tokens_list, FT_VECTOR_SIZE, cache_key)
    st.session_state["X"] = X
    st.session_state["y"] = df_prep["label"].values
    st.session_state["ft_ready"] = True

    st.success(
        f"FastText trained on {len(tokens_list):,} documents · "
        f"vocabulary size: {len(ft_model.wv):,} · embedding dimension: {FT_VECTOR_SIZE} · "
        f"training time: {train_time:.1f}s"
    )
    st.caption(f"Document embedding matrix: {X.shape[0]:,} documents × {X.shape[1]} dimensions.")


# =========================================================
# PAGE 5 - MODEL BUILDING (SVM + Logistic Regression)
# =========================================================
def page5():
    st.subheader("Building Models: SVM & Logistic Regression")

    if not st.session_state.get("ft_ready"):
        st.warning("Please visit the **Text Representation** tab first to generate document embeddings.")
        return

    X = st.session_state["X"]
    y = st.session_state["y"]

    c1, c2 = st.columns(2)
    test_size = c1.slider("Test set size", 0.1, 0.4, 0.2, step=0.05)
    random_state = c2.number_input("Random seed", value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    t0 = time.time()
    log_reg, svm = train_models(X_train, y_train, random_state)
    train_time = time.time() - t0

    st.session_state["log_reg"] = log_reg
    st.session_state["svm"] = svm
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["models_ready"] = True

    st.success(
        f"Trained on {len(X_train):,} rows, testing on {len(X_test):,} rows. "
        f"Training time: {train_time:.2f}s"
    )

    st.markdown("""
    - **Logistic Regression**: a linear probabilistic classifier, used here as a
      fast, interpretable baseline.
    - **SVM (LinearSVC)**: a linear support vector machine, chosen over a
      kernel SVM for scalability on this dataset size; it uses `decision_function`
      scores (rather than `predict_proba`) for ROC-AUC, since `LinearSVC` doesn't
      natively output probabilities.
    """)


# =========================================================
# PAGE 6 - MODEL EVALUATION
# =========================================================
def page6():
    st.subheader("Model Evaluation")

    if not st.session_state.get("models_ready"):
        st.warning("Please visit the **Modeling** tab first to train the models.")
        return

    log_reg = st.session_state["log_reg"]
    svm = st.session_state["svm"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]

    res_lr = evaluate_model(log_reg, X_test, y_test)
    res_svm = evaluate_model(svm, X_test, y_test)
    st.session_state["res_lr"] = res_lr
    st.session_state["res_svm"] = res_svm

    tab_a, tab_b = st.tabs(["Logistic Regression", "SVM (LinearSVC)"])

    for tab, res, name in [(tab_a, res_lr, "Logistic Regression"), (tab_b, res_svm, "SVM")]:
        with tab:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{res['accuracy']:.3f}")
            c2.metric("Precision", f"{res['precision']:.3f}")
            c3.metric("Recall", f"{res['recall']:.3f}")
            c4.metric("F1 Score", f"{res['f1']:.3f}")
            c5.metric("ROC-AUC", f"{res['roc_auc']:.3f}")

            cm = res["confusion_matrix"]
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Negative (0)", "Positive (1)"], y=["Negative (0)", "Positive (1)"],
                title=f"{name} — Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("#### ROC Curves")
    fig_roc = go.Figure()
    for res, name in [(res_lr, "Logistic Regression"), (res_svm, "SVM")]:
        fpr, tpr, _ = roc_curve(y_test, res["y_score"])
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC = {res['roc_auc']:.3f})"
        ))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="gray"), name="Random baseline"))
    fig_roc.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)


# =========================================================
# PAGE 7 - MODEL COMPARISON, BEST MODEL & INTERPRETATION
# =========================================================
def page7():
    st.subheader("Model Comparison & Interpretation")

    if not st.session_state.get("res_lr") or not st.session_state.get("res_svm"):
        st.warning("Please visit the **Evaluation** tab first to compute model metrics.")
        return

    res_lr = st.session_state["res_lr"]
    res_svm = st.session_state["res_svm"]

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comp_df = pd.DataFrame({
        "Metric": [m.replace("_", " ").title() for m in metrics],
        "Logistic Regression": [res_lr[m] for m in metrics],
        "SVM": [res_svm[m] for m in metrics],
    })
    st.dataframe(comp_df.style.format({"Logistic Regression": "{:.3f}", "SVM": "{:.3f}"}),
                 use_container_width=True)

    comp_melted = comp_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
    fig = px.bar(comp_melted, x="Metric", y="Score", color="Model", barmode="group",
                 title="Model Comparison Across All Metrics")
    st.plotly_chart(fig, use_container_width=True)

    chosen_metric = st.selectbox(
        "Choose the metric to determine the best model:",
        [m.replace("_", " ").title() for m in metrics],
        index=3  # default: F1
    )
    metric_key = chosen_metric.lower().replace(" ", "_")
    best_model = "Logistic Regression" if res_lr[metric_key] >= res_svm[metric_key] else "SVM"
    best_score = max(res_lr[metric_key], res_svm[metric_key])
    other_score = min(res_lr[metric_key], res_svm[metric_key])

    st.success(f"🏆 **Best model by {chosen_metric}: {best_model}** ({best_score:.3f} vs {other_score:.3f})")

    st.markdown("### Interpretation")
    st.markdown(f"""
    - Based on **{chosen_metric}**, **{best_model}** performs better, scoring
      **{best_score:.3f}** compared to **{other_score:.3f}** for the alternative model.
    - Both models share the same input representation: FastText embeddings averaged
      per document. Since both are linear classifiers operating on the same features,
      their performance differences largely reflect how each handles the decision
      boundary (margin-based for SVM vs. probabilistic for Logistic Regression) rather
      than differences in the underlying text representation.
    - **Recall vs. Precision trade-off**: if recall is notably higher than precision for
      a model, it is catching more positive-sentiment cases but at the cost of more
      false positives — worth considering depending on whether the application prioritizes
      catching all positive sentiment or minimizing false alarms.
    - **Limitations to note in the report**: (1) averaging word vectors discards word
      order, which limits sentiment cues from negation or sarcasm; (2) stopword removal
      was only available for {len(LANG_TO_NLTK)} of the {df_prep['lang'].nunique()} languages
      via nltk, so preprocessing depth varies by language; (3) this app uses a
      language-and-label-stratified *sample* of the full dataset (capped per language) for
      performance reasons, not the complete corpus — results should be described as
      indicative rather than final production benchmarks.
    """)


# =========================================================
# NAVIGATION
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Raw Data", "🧹 Cleaning & Preprocessing", "🔍 EDA",
    "🔤 Representation", "🤖 Modeling", "📈 Evaluation", "🏆 Comparison"
])
with tab1:
    page1()
with tab2:
    page2()
with tab3:
    page3()
with tab4:
    page4()
with tab5:
    page5()
with tab6:
    page6()
with tab7:
    page7()