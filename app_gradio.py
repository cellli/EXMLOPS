import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax

# Carica modello
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def predict_sentiment(text):
    if not text.strip():
        return "Inserisci un testo", 0, {}
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    scores = outputs[0][0].detach().numpy()
    scores = softmax(scores)
    
    labels = ['Negative 😞', 'Neutral 😐', 'Positive 😊']
    ranking = np.argsort(scores)[::-1]
    top_label = labels[ranking[0]]
    confidence = float(scores[ranking[0]])
    
    scores_dict = {labels[i]: float(scores[i]) for i in range(3)}
    
    return top_label, f"{confidence:.2%}", scores_dict

# Interfaccia Gradio
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Inserisci un tweet o commento...", label="Testo"),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Confidenza"),
        gr.JSON(label="Dettaglio Score")
    ],
    title="🚀 MachineInnovators - Sentiment Analysis",
    description="Sistema di analisi del sentiment basato su RoBERTa. Inserisci un testo per analizzarlo.",
    examples=[
        ["I absolutely love this product! Best purchase ever!"],
        ["Terrible customer service, never buying again."],
        ["The delivery was on time, nothing special."]
    ]
)

demo.launch()
