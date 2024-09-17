from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch

# BERT for Text Classification
def classify_text_bert(text):
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # Prepare input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

# T5 for Text Summarization
def summarize_text_t5(text):
    # Load pre-trained T5 model and tokenizer
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Prepare input text
    inputs = tokenizer("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Example usage
if __name__ == "__main__":
    # Example text for BERT
    review_text = "I love this product! It is amazing and worth every penny."
    class_label = classify_text_bert(review_text)
    print(f"Predicted Class Label: {class_label}")

    # Example text for T5
    long_text = """
    The United Nations is an international organization founded in 1945. It is currently made up of 193 Member States. The mission and work of the United Nations are guided by the purposes and principles contained in its founding Charter. 
    The United Nations was established after the Second World War by the Allied powers, as a successor to the ineffective League of Nations. The organization's objectives are to maintain international peace and security, promote human rights, foster social and economic development, and coordinate international cooperation.
    """
    summary = summarize_text_t5(long_text)
    print(f"Summary: {summary}")
