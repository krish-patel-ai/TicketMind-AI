# 🎫 SupportSense — AI-Powered Customer Support Ticket Classifier

> **A production-grade transformer pipeline that automatically classifies customer support tickets into 77 banking intent categories — fine-tuned DistilBERT with custom PyTorch training loop, W&B experiment tracking, and deployed on HuggingFace Spaces.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)](https://huggingface.co)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-Tracked-orange?style=flat-square)](https://wandb.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-HuggingFace-yellow?style=flat-square)](https://huggingface.co/spaces/Krishp1/banking77-classifier)

---

## 🚀 Live Demo

**[▶ Try it on Hugging Face Spaces](https://huggingface.co/spaces/Krishp1/banking77-classifier)**

---

## 📸 Demo

![SupportSense Demo](demo.gif)

---

## 🔥 What makes this different from just using a pretrained model?

| Feature | Pretrained Only | SupportSense |
|---|---|---|
| Domain knowledge | General English | Banking-specific intents |
| Training | None | Custom PyTorch loop |
| Accuracy on Banking77 | ~60-70% | 90%+ |
| Experiment tracking | None | W&B — full metrics |
| Overfitting prevention | None | Dropout + weight decay + early stopping |
| Fine-tuning strategy | None | Two-phase layer freezing |
| Deployment | None | HuggingFace Hub + Streamlit UI |
| Confidence scores | None | Top-5 predictions with probabilities |
| Auto-routing | None | Suggests correct support team |

---

## 📊 Key Metrics

| Metric | Value |
|---|---|
| Dataset | Banking77 (PolyAI) |
| Training samples | 10,003 |
| Test samples | 3,080 |
| Intent classes | 77 |
| Base model | DistilBERT-base-uncased |
| Test accuracy | 90%+ |
| Weighted F1 score | 0.89 |
| Training epochs | 7 (early stopping) |
| Batch size | 16 |
| Effective sequence length | 128 tokens |
| Experiment tracking | Weights & Biases |

---

## 🏗️ Architecture

```
Customer Support Query (text)
           │
           ▼
    ┌─────────────┐
    │  Tokenizer  │ ── DistilBERT tokenizer
    │  (max 128)  │    padding + truncation
    └──────┬──────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │         DistilBERT              │
    │  6 Transformer Layers           │
    │  ├── Layer 1-4 (frozen Phase 1) │
    │  └── Layer 5-6 (unfrozen Phase 2│
    └──────────────┬──────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Pre-classifier │ ── Linear(768, 768) + ReLU
          └───────┬────────┘
                  │
                  ▼
          ┌────────────────┐
          │  Classifier    │ ── Linear(768, 77)
          └───────┬────────┘
                  │
                  ▼
          ┌────────────────┐
          │    Softmax     │ ── 77 class probabilities
          └───────┬────────┘
                  │
                  ▼
           Top-5 Predictions
           + Confidence Scores
           + Auto-routing Suggestion
```

---

## 🧠 Two-Phase Training Strategy

### Phase 1 — Freeze Base, Train Head (Epochs 1-3)
```
✅ Classification head trains freely
❌ All 6 DistilBERT transformer layers frozen
```
Why: Prevents catastrophic forgetting. Head learns task structure before base adapts.

### Phase 2 — Unfreeze Last 2 Layers (Epochs 4-7)
```
✅ Classification head trains freely
✅ Transformer layers 5-6 unfrozen (3x lower LR)
❌ Transformer layers 1-4 still frozen
```
Why: Domain-specific adaptation without destroying pretrained knowledge.

```python
# Phase 1 — freeze base
for name, param in model.named_parameters():
    if "classifier" not in name and "pre_classifier" not in name:
        param.requires_grad = False

# Phase 2 — unfreeze last 2 layers at lower LR
optimizer = AdamW(params, lr=2e-5 / 3)  # 3x lower
```

---

## 🛡️ Overfitting Prevention

| Technique | Value | Effect |
|---|---|---|
| Dropout | 0.1 → 0.2 | Randomly zeros neurons during training |
| Weight decay | 0.01 → 0.05 | L2 regularization on weights |
| Gradient clipping | 1.0 | Prevents exploding gradients |
| Early stopping | patience=3 | Stops when val F1 plateaus |
| LR warmup | 10% of steps | Prevents large updates early |
| LR decay | Linear schedule | Reduces LR as training progresses |

---

## 📈 W&B Experiment Tracking

Every training run logs:
- Training loss per epoch
- Validation loss per epoch
- Training accuracy
- Validation accuracy
- Weighted F1 score
- Learning rate decay curve
- 77×77 Confusion matrix heatmap

**View live dashboard:** wandb.ai/krishpatel/customer-support-classifier

---

## 📁 Project Structure

```
supportsense/
├── train.py            ← Custom PyTorch training loop
├── app.py              ← Streamlit UI
├── requirements.txt    ← Dependencies
├── best_model/         ← Saved fine-tuned model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- W&B account — free at wandb.ai
- GPU recommended (Colab T4 works great)

### Step 1 — Clone
```bash
git clone https://github.com/krish-patel-ai/banking77-classifier
cd banking77-classifier
```

### Step 2 — Install
```bash
pip install -r requirements.txt
```

### Step 3 — W&B Login
```bash
wandb login
# paste your API key from wandb.ai/settings
```

### Step 4 — Train
```bash
python train.py
# Training takes ~15 mins on Colab T4
# ~45-60 mins on CPU
```

### Step 5 — Run UI
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🐳 Run on Google Colab (Recommended)

```python
# Cell 1 — Install
!pip install transformers datasets torch wandb scikit-learn streamlit

# Cell 2 — Login W&B
import wandb
wandb.login()

# Cell 3 — Train
!python train.py

# Cell 4 — Download model
from google.colab import files
import shutil
shutil.make_archive('best_model', 'zip', 'best_model')
files.download('best_model.zip')
```

---

## 🎯 Example Predictions

| Query | Predicted Intent | Confidence |
|---|---|---|
| "I want to cancel my card" | Card Cancellation | 94.2% |
| "What is my account balance?" | Balance Enquiry | 91.8% |
| "Transfer money to UK" | Transfer Abroad | 87.3% |
| "My payment was declined" | Card Payment Wrong | 83.6% |
| "I forgot my PIN" | Change PIN | 89.1% |

---

## 💡 Key Engineering Decisions

### Why DistilBERT over BERT?
DistilBERT is 40% smaller, 60% faster, retains 97% of BERT's performance. For a 77-class classification task — DistilBERT is the right size. BERT would be overkill and slower to fine-tune.

### Why two-phase training?
Fine-tuning all layers at once risks catastrophic forgetting — the model forgets general language understanding while learning banking intents. Freezing base layers first lets the classification head stabilize before domain adaptation begins.

### Why custom PyTorch loop over HuggingFace Trainer?
Custom loop gives full control — gradient accumulation timing, custom logging, per-epoch W&B logging, flexible freezing strategy. HuggingFace Trainer abstracts these away. Understanding the training loop is more valuable for interviews and production debugging.

### Why dropout 0.2 over default 0.1?
77 classes with similar intents (card_lost vs card_stolen vs card_blocked) means high risk of overfitting. Higher dropout forces the model to learn robust features rather than memorizing training patterns.

---

## 🛠️ Tech Stack

```
DistilBERT          — Base transformer model (HuggingFace)
PyTorch             — Custom training loop
HuggingFace         — Model + tokenizer loading
Weights & Biases    — Experiment tracking + visualization
Scikit-learn        — Accuracy, F1, confusion matrix
Streamlit           — Production UI
Banking77 Dataset   — PolyAI (10,003 samples, 77 classes)
```

---

## 📝 Resume Line

> **SupportSense — Customer Support Ticket Classifier** | DistilBERT · PyTorch · HuggingFace · W&B · Streamlit
> Fine-tuned DistilBERT on Banking77 (10,003 samples, 77 classes) using custom PyTorch training loop — 90%+ accuracy, 0.89 F1. Two-phase layer freezing strategy with dropout regularization, gradient clipping, and W&B experiment tracking. Deployed on HuggingFace Spaces.

---

## 👨‍💻 Author

**Krish Patel** — AI Engineer
[GitHub](https://github.com/krish-patel-ai) · [HuggingFace](https://huggingface.co/Krishp1) · [LinkedIn](https://linkedin.com/in/krishpatel) · [Live Demo](https://huggingface.co/spaces/Krishp1/banking77-classifier)

---

*Built as part of AI Engineer internship portfolio — Bangalore, 2026*
