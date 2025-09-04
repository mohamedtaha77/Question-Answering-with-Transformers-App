# ❓ Question Answering Web App - Elevvo NLP Internship

This is a clean and interactive **Question Answering (QA) App** built with **Streamlit**.  
It extracts answers from a given paragraph using **Transformer-based models**.

Three models are implemented and fully deployed in the UI:
- **DistilBERT (HuggingFace Transformers)**
- **BERT Base Uncased**
- **BERT Large Uncased**

---

## 📌 Features

✅ Extracts answers from context paragraphs using Transformers  
✅ Supports **DistilBERT**, **BERT-base**, and **BERT-large** models  
✅ Uses HuggingFace’s `pipeline("question-answering")`  
✅ Streamlined interface for paragraph and question input  
✅ Displays extracted **answer**, **score**, and **answer span**  
✅ Fully interactive **Streamlit UI** with dropdown model selection  
✅ Optimized loading using `torch.no_grad()` and caching  
✅ Complete training and inference workflow documented

---

## 🔗 Live Demo

Try the app live here:  
👉 [https://question-answering-with-transformers-77.streamlit.app/](https://question-answering-with-transformers-77.streamlit.app/)

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app for transformer-based QA |
| `qa_distilbert_train_and_save.py` | Notebook for fine-tuning and saving DistilBERT model |
| `qa_bertbase_train_and_save.py` | Notebook for fine-tuning and saving BERT Base model |
| `qa_bertlarge_train_and_save.py` | Notebook for fine-tuning and saving BERT Large model |
| `requirements.txt` | All required Python packages |
| `model/` | Contains saved model files and tokenizer checkpoints |

---

## 🚀 How to Run Locally

1. **Clone the repo**:

```bash
git clone https://github.com/yourusername/question-answering-transformers.git
cd question-answering-transformers
```

2. **(Optional) Create a virtual environment**:

```bash
python -m venv venv
venv\Scripts\activate  # on Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run the app**:

```bash
streamlit run app.py
```

---

## 🌍 Deployment (Optional)

To deploy on **Streamlit Cloud**:

- Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
- Connect your GitHub repo
- Set the main file to: `app.py`
- Make sure your models are in the `model/` folder or downloaded via HuggingFace inside the script

---

## 📓 Notebook Workflow

> [`qa_*_train_and_save.py`](./)

Due to resource constraints, the training for each model was conducted in **separate notebooks**, ensuring stability and avoiding runtime crashes. Each notebook follows the same workflow:

- Load pre-trained model from HuggingFace (`distilbert-base-uncased`, `bert-base-uncased`, `bert-large-uncased`)
- Preprocess SQuAD-like QA data (context, question, answer)
- Fine-tune using HuggingFace Trainer API
- Save model weights and tokenizer locally for deployment
- Load for inference using the QA pipeline

---

## ⚠️ Final Remarks

This task demonstrates the power of pre-trained Transformer models for QA. While all models deliver impressive performance, **BERT Large** offers superior accuracy at the cost of speed and size. The app provides flexibility to explore all three models, showcasing trade-offs between inference time and model complexity. Future improvements could include domain-specific finetuning and support for long contexts.

Web app link: [https://question-answering-with-transformers-77.streamlit.app/](https://question-answering-with-transformers-77.streamlit.app/)