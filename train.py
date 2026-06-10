import os
import torch
import wandb
import numpy as np
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ─────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ── WANDB ──────────────────────────────
wandb.init(
    project="customer-support-classifier",
    config={
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_len": MAX_LEN
    }
)

# ── DATASET ────────────────────────────
print("Loading Banking77 dataset...")
dataset    = load_dataset("banking77")
train_data = dataset["train"]
test_data  = dataset["test"]

num_labels = 77
print(f"Train size: {len(train_data)} | Test size: {len(test_data)}")

# ── TOKENIZER ──────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# ── CUSTOM DATASET CLASS ───────────────
class BankingDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data      = data
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text  = self.data[idx]["text"]
        label = self.data[idx]["label"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(label, dtype=torch.long)
        }

# ── DATALOADERS ────────────────────────
train_dataset = BankingDataset(train_data, tokenizer, MAX_LEN)
test_dataset  = BankingDataset(test_data, tokenizer, MAX_LEN)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── MODEL ──────────────────────────────
print("Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)
model.to(DEVICE)

# ── OPTIMIZER + SCHEDULER ──────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps    = len(train_loader) * EPOCHS
scheduler      = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# ── LOSS ───────────────────────────────
criterion = torch.nn.CrossEntropyLoss()

# ── TRAINING LOOP ──────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss    = 0
    all_preds     = []
    all_labels    = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

        loss    = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, acc, f1

# ── EVALUATION LOOP ────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            loss    = criterion(logits, labels)
            total_loss += loss.item()

            preds   = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, acc, f1, all_preds, all_labels

# ── RUN TRAINING ───────────────────────
print("\nStarting training...")
best_f1 = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc, train_f1 = train_epoch(
        model, train_loader, optimizer, scheduler, criterion
    )
    val_loss, val_acc, val_f1, preds, labels = evaluate(
        model, test_loader, criterion
    )

    print(f"Train — Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Val   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

    wandb.log({
        "epoch":       epoch + 1,
        "train_loss":  train_loss,
        "train_acc":   train_acc,
        "train_f1":    train_f1,
        "val_loss":    val_loss,
        "val_acc":     val_acc,
        "val_f1":      val_f1
    })

    if val_f1 > best_f1:
        best_f1 = val_f1
        model.save_pretrained("best_model")
        tokenizer.save_pretrained("best_model")
        print(f"✅ Best model saved! F1: {best_f1:.4f}")

# ── CONFUSION MATRIX ───────────────────
print("\nGenerating confusion matrix...")
_, _, _, final_preds, final_labels = evaluate(model, test_loader, criterion)

cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — Banking77")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("✅ Saved confusion_matrix.png")

wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
wandb.finish()

print(f"\n🎉 Training complete! Best F1: {best_f1:.4f}")