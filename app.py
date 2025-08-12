import streamlit as st
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification

# Load model and tokenizer
@st.cache_resource  # cache so it doesn't reload on every interaction
def load_model():
    model_path = "roberta_model"  # your saved model folder
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

st.title("Transformer Hate Speech Classifier")

# Input text box
user_input = st.text_area("Enter text to classify:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize input
        inputs = tokenizer(
            user_input,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        # Get predicted label and confidence
        pred_label_id = probs.argmax()
        confidence = probs[pred_label_id]
        label_map = model.config.id2label
        pred_label = label_map[pred_label_id]

        st.write(f"### Prediction: {pred_label}")
        st.write(f"Confidence: {confidence:.2%}")

        # Optional: Show probability distribution for all classes
        st.write("#### Class probabilities:")
        for i, prob in enumerate(probs):
            st.write(f"{label_map[i]}: {prob:.2%}")
