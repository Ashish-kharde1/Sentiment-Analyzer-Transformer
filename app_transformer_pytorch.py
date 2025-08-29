import streamlit as st
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re

model_path = "assets/sentiment_distilert_pytorch"
st.set_page_config(page_title = "Movie Review Sentiment Analyzer (PyTorch)", page_icon="🎬")


def load_model_and_tokenizer():
    try:
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.error(f"Please make sure the model is saved in the '{model_path}' directory.")
        return None,None
    
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    return text

def predict_sentiment(review, model, tokenizer):
    if not review.strip():
        return None,None
    
    cleaned_text = clean_text(review)
    inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding = True, max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()[0]

    prediction = np.argmax(probabilities)
    confidence = probabilities[prediction]

    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, confidence

# --- Streamlit UI ---
st.title("🎬 Movie Review Sentiment Analyzer (Upgraded with BERT/PyTorch)")
st.markdown(
    "Enter a movie review to classify its sentiment as **Positive** or **Negative**. "
    "This version uses a fine-tuned **DistilBERT** model with **PyTorch**."
)

model, tokenizer = load_model_and_tokenizer()

if model and tokenizer:
    review_input = st.text_area(
        "Enter your review here:",
        height=150,
        placeholder="e.g., 'This movie was absolutely fantastic! The acting was superb and the plot was gripping.'"
    )

    if st.button("Analyze Sentiment", type="primary"):
        with st.spinner("Analyzing with the power of Transformers..."):
            sentiment, score = predict_sentiment(review_input, model, tokenizer)
        
        if sentiment:
            st.write("---")
            st.subheader("Analysis Result")
            if sentiment == "Positive":
                st.success(f"Sentiment: {sentiment} 👍")
            else:
                st.error(f"Sentiment: {sentiment} 👎")
            
            st.progress(float(score))
            st.metric(label="Confidence Score", value=f"{score:.2%}")
        else:
            st.warning("Please enter a review to analyze.")
else:
    st.error("The upgraded PyTorch model could not be loaded. Please run the training notebook first.")