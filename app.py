import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch

# Load models and tokenizers
def load_models():
    bert_model_name = 'bert-base-uncased'
    t5_model_name = 't5-small'

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)

    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    return bert_tokenizer, bert_model, t5_tokenizer, t5_model

bert_tokenizer, bert_model, t5_tokenizer, t5_model = load_models()

# BERT for Text Classification
def classify_text_bert(text):
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# T5 for Text Summarization
def summarize_text_t5(text):
    inputs = t5_tokenizer("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        summary_ids = t5_model.generate(inputs['input_ids'], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.title("Text Processing with BERT and T5")

st.sidebar.header("Select Task")
task = st.sidebar.selectbox("Choose a task", ["Text Classification (BERT)", "Text Summarization (T5)"])

if task == "Text Classification (BERT)":
    st.header("Text Classification")
    input_text = st.text_area("Enter text for classification", "")
    if st.button("Classify"):
        if input_text:
            class_label = classify_text_bert(input_text)
            st.write(f"Predicted Class Label: {class_label}")
        else:
            st.error("Please enter some text.")

elif task == "Text Summarization (T5)":
    st.header("Text Summarization")
    input_text = st.text_area("Enter text for summarization", "")
    if st.button("Summarize"):
        if input_text:
            summary = summarize_text_t5(input_text)
            st.write("Summary:")
            st.write(summary)
        else:
            st.error("Please enter some text.")
