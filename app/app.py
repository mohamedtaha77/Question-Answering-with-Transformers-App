# app.py

import os
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# ========== Config ==========
st.set_page_config(page_title="🧠 QA App - Elevvo Task 6", layout="centered")
st.title("🧠 Question Answering with Transformers")
st.subheader("Fine-tuned DistilBERT | Elevvo Internship — Task 6")

# ---------- Model selector (NEW) ----------
# Map visible names to local folders (edit if you rename folders)
MODEL_PATHS = {
    "DistilBERT (base-uncased)": "./qa_model_distilbert-base-uncased",
    "BERT (base-uncased)": "./qa_model_bert-base-uncased",
    "RoBERTa (base)": "./qa_model_roberta-base",
}

# keep previously selected model across reruns
if "selected_model_key" not in st.session_state:
    st.session_state.selected_model_key = "DistilBERT (base-uncased)"

st.markdown("### 🧩 Choose Model")
selected_model_key = st.selectbox(
    "Pick a fine-tuned model to run:",
    options=list(MODEL_PATHS.keys()),
    index=list(MODEL_PATHS.keys()).index(st.session_state.selected_model_key)
)
st.session_state.selected_model_key = selected_model_key

# Resolve absolute model path and device
model_path = os.path.abspath(MODEL_PATHS[selected_model_key])
device = 0 if torch.cuda.is_available() else -1

st.caption(f"Using **{selected_model_key}** from `{model_path}` | Device: "
           f"{'CUDA' if device == 0 else 'CPU'}")

# ========== Load QA Pipeline ==========
@st.cache_resource(show_spinner=False)
def load_pipeline(model_path: str, device: int):
    """
    Cache one pipeline per (model_path, device).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only=True)
    return pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

# try to load; surface a helpful error if model files are missing
qa_pipeline = None
try:
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model folder not found: {model_path}")
    qa_pipeline = load_pipeline(model_path, device)
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"⚠️ Could not load the selected model. {e}")
    st.stop()

# ========== Test Sample ==========
with st.expander("🧪 Try a Sample (Click to Fill)"):
    if st.button("Load Example"):
        st.session_state.context = (
            "The Nile is the longest river in the world. It flows through Egypt and Sudan. "
            "It has been an important source of life and transportation for thousands of years."
        )
        st.session_state.question = "Where does the Nile flow?"

# ========== Input ==========
st.markdown("### 🔍 Ask a Question")
context = st.text_area("📄 Context", height=200, key="context", placeholder="Paste or load a paragraph...")
question = st.text_input("❓ Question", key="question", placeholder="Type your question here...")

if st.button("Get Answer") and context.strip() and question.strip():
    with st.spinner("Generating answer..."):
        result = qa_pipeline(question=question, context=context)
        st.success("✅ Answer generated!")
        st.write(f"**Answer:** {result['answer']}")
        st.write(f"**Confidence Score:** {round(result['score'] * 100, 2)}%")

# ========== Metrics ==========
st.markdown("---")
st.markdown("## 📊 Model Performance Summary")

@st.cache_data(show_spinner=False)
def load_comparison_data():
    csv_path = os.path.abspath("./results/model_comparison_summary.csv")
    return pd.read_csv(csv_path)

try:
    df = load_comparison_data()

    def get_best(df, col, ascending=False):
        return df.sort_values(by=col, ascending=ascending).iloc[0]

    best_f1 = get_best(df, 'f1')
    best_em = get_best(df, 'exact_match')
    best_time = get_best(df, 'load_time_s', ascending=True)
    best_score = get_best(df, 'avg_pipeline_score')

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔹 Highest F1 Score", best_f1["model"], f"{best_f1['f1']:.2f}")
        st.metric("🚀 Fastest Load Time", best_time["model"], f"{best_time['load_time_s']}s")
    with col2:
        st.metric("🎯 Highest Exact Match", best_em["model"], f"{best_em['exact_match']:.2f}")
        st.metric("⚡ Most Efficient (Avg. Score)", best_score["model"], f"{best_score['avg_pipeline_score']:.2f}")

    st.markdown("### 📊 Comparison Chart")

    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df.set_index("model")[["f1", "exact_match", "avg_pipeline_score"]]
    df_plot.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("QA Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim([0, 100])
    ax.legend(loc='lower right')
    st.pyplot(fig)

    with st.expander("🧾 Raw Table"):
        st.dataframe(df, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Error loading metrics: {e}")

# ========== Footer ==========
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>Made with ❤️ for Elevvo Internship Task 6</div>",
    unsafe_allow_html=True
)
