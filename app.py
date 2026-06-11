import torch
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset

# ── LABELS ─────────────────────────────
@st.cache_resource
def load_labels():
    return [
        "activate_my_card", "age_limit", "apple_pay_or_google_pay",
        "atm_support", "automatic_top_up", "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
        "cancel_transfer", "card_about_to_expire", "card_acceptance",
        "card_arrival", "card_delivery_estimate", "card_linking",
        "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
        "contactless_not_working", "country_support", "declined_card_payment",
        "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
        "disposable_card_limits", "edit_personal_details", "exchange_charge",
        "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
        "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card",
        "get_physical_card", "getting_spare_card", "getting_virtual_card",
        "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
        "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal",
        "pending_top_up", "pending_transfer", "pin_blocked",
        "receiving_money", "Refund_not_showing_up", "request_refund",
        "reverted_card_payment", "supported_cards_and_currencies", "terminate_account",
        "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque",
        "top_up_failed", "top_up_limits", "top_up_reverted",
        "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
        "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
        "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
        "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
        "why_verify_identity", "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal"
    ]
# ── MODEL ──────────────────────────────

@st.cache_resource
@st.cache_resource
def load_model():
    print("STEP 1: Starting")

    tokenizer = DistilBertTokenizer.from_pretrained(
        "Krishp1/ticketmind-model",
        subfolder="best_model"
    )

    print("STEP 2: Tokenizer loaded")

    model = DistilBertForSequenceClassification.from_pretrained(
        "Krishp1/ticketmind-model",
        subfolder="best_model"
    )

    print("STEP 3: Model loaded")

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
    page_icon="",
    layout="centered"
)

st.title("Customer Support Ticket Classifier")
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
st.caption("Fine-tuned DistilBERT | Banking77")
