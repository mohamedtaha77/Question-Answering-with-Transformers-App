import os
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import random

# ========== Config ==========
st.set_page_config(page_title="🧠 QA App", layout="centered")
st.title("🧠 Question Answering with Transformers")

# ---------- Model selector ----------
MODEL_PATHS = {
    "DistilBERT": "./qa_model_distilbert-base-uncased",
    "BERT": "./qa_model_bert-base-uncased",
    "RoBERTa": "./qa_model_roberta-base",
}

if "selected_model_key" not in st.session_state:
    st.session_state.selected_model_key = "DistilBERT"

if st.session_state.selected_model_key not in MODEL_PATHS:
    st.session_state.selected_model_key = "DistilBERT"  # Default to DistilBERT if key doesn't exist

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
        # Adding 10 mock context-question pairs to simulate random selection
        sample_data = [
            {"context": "The Nile is the longest river in the world. It flows through Egypt and Sudan.", "question": "Where does the Nile flow?"},
            {"context": "Mount Everest is the tallest mountain in the world. Located in the Himalayas.", "question": "What is the height of Mount Everest?"},
            {"context": "The Great Wall of China is one of the most famous landmarks in the world.", "question": "How long is the Great Wall of China?"},
            {"context": "The Eiffel Tower is one of the most recognizable structures in the world.", "question": "Where is the Eiffel Tower located?"},
            {"context": "The Amazon rainforest is located in South America.", "question": "Where is the Amazon rainforest?"},
            {"context": "The Pacific Ocean is the largest and deepest ocean on Earth.", "question": "What is the largest ocean?"},
            {"context": "The human brain is the central organ of the nervous system.", "question": "What is the human brain responsible for?"},
            {"context": "Beijing is the capital of China.", "question": "What is the capital of China?"},
            {"context": "The Mona Lisa is a famous painting by Leonardo da Vinci.", "question": "Who painted the Mona Lisa?"},
            {"context": "The moon orbits the Earth, making one full orbit every 27.3 days.", "question": "How long does it take for the moon to orbit the Earth?"}
        ]

        # Select 10 random samples
        selected_samples = random.sample(sample_data, 10)

        # Store the randomly selected samples in session state
        st.session_state.samples = selected_samples

        # Display the selected samples
        for idx, sample in enumerate(st.session_state.samples):
            st.write(f"**Sample {idx+1}:**")
            st.write(f"**Context:** {sample['context']}")
            st.write(f"**Question:** {sample['question']}")

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
    # Replace with the correct CSV path to include the correct metrics in the CSV
    csv_path = os.path.abspath("./results/model_comparison_summary.csv")
    return pd.read_csv(csv_path)

try:
    df = load_comparison_data()

    def get_best(df, col, ascending=False):
        return df.sort_values(by=col, ascending=ascending).iloc[0]

    # Collecting the metrics from the CSV
    best_loss = get_best(df, 'Eval Loss')
    best_f1 = get_best(df, 'Eval F1')
    best_em = get_best(df, 'Eval EM')
    best_time = get_best(df, 'Runtime (s)', ascending=True)
    best_samples = get_best(df, 'Samples/s', ascending=False)
    best_steps = get_best(df, 'Steps/s', ascending=False)

    # Displaying the metrics on the Streamlit app
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔹 Lowest Eval Loss", best_loss["Model"], f"{best_loss['Eval Loss']:.2f}")
        st.metric("🚀 Fastest Runtime", best_time["Model"], f"{best_time['Runtime (s)']}s")
        st.metric("🎯 Highest F1 Score", best_f1["Model"], f"{best_f1['Eval F1']:.2f}")

    with col2:
        st.metric("⚡ Highest Exact Match", best_em["Model"], f"{best_em['Eval EM']:.2f}")
        st.metric("⚡ Best Samples/s", best_samples["Model"], f"{best_samples['Samples/s']:.2f}")
        st.metric("⚡ Best Steps/s", best_steps["Model"], f"{best_steps['Steps/s']:.2f}")

    st.markdown("### 📊 Comparison Chart")

    # Plotting a comparison chart for model metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df.set_index("Model")[["Eval F1", "Eval EM", "Samples/s", "Steps/s"]]
    df_plot.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("QA Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim([0, max(df["Samples/s"].max(), 100)])
    ax.legend(loc='lower right')
    st.pyplot(fig)
    plt.xticks(rotation=45, ha='right')


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
