# app.py

import os
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# ========== Config ==========
st.set_page_config(page_title="🧠 QA App", layout="centered")
st.title("🧠 Question Answering with Transformers")

# ---------- Model selector ----------
MODEL_PATHS = {
    "DistilBERT (base-uncased)": "./qa_model_distilbert-base-uncased",
    "BERT (base-uncased)": "./qa_model_bert-base-uncased",
    "RoBERTa (base)": "./qa_model_roberta-base",
}

if "selected_model_key" not in st.session_state:
    st.session_state.selected_model_key = "DistilBERT (base-uncased)"

st.markdown("### 🧩 Choose Model")
selected_model_key = st.selectbox(
    "Pick a fine-tuned model to run:",
    options=list(MODEL_PATHS.keys()),
    index=list(MODEL_PATHS.keys()).index(st.session_state.selected_model_key)
)
st.session_state.selected_model_key = selected_model_key

model_path = os.path.abspath(MODEL_PATHS[selected_model_key])
device = 0 if torch.cuda.is_available() else -1

# ========== Load QA Pipeline ==========
@st.cache_resource(show_spinner=False)
def load_pipeline(model_path: str, device: int):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only=True)
    return pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

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

# ======== Result Section ========
if st.button("Get Answer") and context.strip() and question.strip():
    with st.spinner("Generating answer..."):
        result = qa_pipeline(question=question, context=context)
        ans_text = result["answer"]
        conf_pct = result["score"] * 100

    col_a, col_b = st.columns([2, 1], gap="medium")

    with col_a:
        st.markdown(
            f"""
            <div style="
                background:#0f5132;
                border:1px solid #0f5132;
                color:#d1e7dd;
                padding:14px 16px;
                border-radius:10px;
                line-height:1.5;
            ">
                <div style="opacity:.9; font-weight:600; margin-bottom:6px;">Answer</div>
                <div style="font-size:1.05rem; font-weight:600;">{ans_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            f"""
            <div style="
                background:#052e16;
                border:1px solid #14532d;
                color:#d1e7dd;
                padding:14px 16px;
                border-radius:10px;
                text-align:center;
            ">
                <div style="opacity:.9; font-weight:600; margin-bottom:6px;">Confidence</div>
                <div style="font-size:1.3rem; font-weight:800;">{conf_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

    best_loss = get_best(df, 'Eval Loss')
    best_f1 = get_best(df, 'Eval F1')
    best_em = get_best(df, 'Eval EM')
    best_time = get_best(df, 'Runtime (s)', ascending=True)
    best_samples = get_best(df, 'Samples/s', ascending=False)
    best_steps = get_best(df, 'Steps/s', ascending=False)
    best_epoch = get_best(df, 'Epoch')

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔹 Lowest Eval Loss", best_loss["Model"], f"{best_loss['Eval Loss']:.2f}")
        st.metric("🚀 Fastest Runtime", best_time["Model"], f"{best_time['Runtime (s)']}s")
        st.metric("🎯 Highest F1 Score", best_f1["Model"], f"{best_f1['Eval F1']:.2f}")

    with col2:
        st.metric("⚡ Highest Exact Match", best_em["Model"], f"{best_em['Eval EM']:.2f}")
        st.metric("⚡ Best Samples/s", best_samples["Model"], f"{best_samples['Samples/s']:.2f}")
        st.metric("⚡ Best Steps/s", best_steps["Model"], f"{best_steps['Steps/s']:.2f}")
        st.metric("⚡ Best Epoch", best_epoch["Model"], f"{best_epoch['Epoch']:.2f}")

    st.markdown("### 📊 Comparison Chart")

    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df.set_index("Model")[["Eval F1", "Eval EM", "Samples/s", "Steps/s"]]
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
    "<div style='text-align: center;'>Made with ❤️</div>",
    unsafe_allow_html=True
)
