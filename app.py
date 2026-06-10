import torch
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset

# ── LABELS ─────────────────────────────
@st.cache_resource
def load_labels():
    dataset = load_dataset("banking77")
    return dataset["train"].features["label"].names

# ── MODEL ──────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("best_model")
    model     = DistilBertForSequenceClassification.from_pretrained("best_model")
    model.eval()
    return tokenizer, model

# ── PREDICT ────────────────────────────
def predict(text, tokenizer, model, labels):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits  = outputs.logits
        probs   = torch.softmax(logits, dim=1)
        top5    = torch.topk(probs, 5)

    results = []
    for score, idx in zip(top5.values[0], top5.indices[0]):
        results.append({
            "category":   labels[idx.item()].replace("_", " ").title(),
            "confidence": round(score.item() * 100, 2)
        })

    return results

# ── UI ─────────────────────────────────
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

st.title("🎫 Customer Support Ticket Classifier")
st.markdown("Powered by fine-tuned DistilBERT on Banking77 dataset")
st.markdown("---")

labels            = load_labels()
tokenizer, model  = load_model()

ticket = st.text_area(
    "Enter your support ticket:",
    placeholder="e.g. My payment failed but money was deducted from my account",
    height=120
)

if st.button("Classify Ticket", type="primary"):
    if ticket.strip():
        with st.spinner("Analyzing..."):
            results = predict(ticket, tokenizer, model, labels)

        st.markdown("### Results:")
        top = results[0]
        st.success(f"**Primary Category: {top['category']}** ({top['confidence']}% confidence)")

        st.markdown("**Top 5 predictions:**")
        for r in results:
            st.progress(
                r["confidence"] / 100,
                text=f"{r['category']} — {r['confidence']}%"
            )
    else:
        st.warning("Please enter a ticket first.")

st.markdown("---")
st.caption("Fine-tuned DistilBERT | Banking77 | Built for AI Engineer Portfolio")