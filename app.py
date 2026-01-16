"""
MachineInnovators Inc. - Sentiment Analysis Module
==================================================
Modulo principale per l'analisi del sentiment basato su RoBERTa.

Utilizzo:
    from app import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    result = analyzer.predict("Your text here")
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

class SentimentAnalyzer:
    """
    Classe per l'analisi del sentiment utilizzando il modello RoBERTa pre-addestrato.
    
    Attributes:
        model_name (str): Nome del modello HuggingFace
        device (str): Dispositivo di computazione (cuda/cpu)
        labels (list): Lista delle etichette di classificazione
    """
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Inizializza il modello e il tokenizer.
        
        Args:
            model_name (str): Nome del modello HuggingFace da caricare
        """
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Caricamento modello {model_name} su {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Modalità evaluation
        
        self.labels = ['Negative', 'Neutral', 'Positive']
        print(f"✅ Modello caricato con successo!")

    def predict(self, text):
        """
        Analizza il sentiment di un testo.
        
        Args:
            text (str): Testo da analizzare
            
        Returns:
            dict: Dizionario con 'sentiment', 'confidence' e 'scores'
        """
        # Tokenizzazione
        encoded_input = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=128, 
            padding=True
        ).to(self.device)

        # Inferenza
        with torch.no_grad():
            output = self.model(**encoded_input)

        # Calcolo probabilità
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)

        # Trova la classe con il punteggio più alto
        ranking = np.argsort(scores)[::-1]
        top_label = self.labels[ranking[0]]
        confidence = scores[ranking[0]]

        return {
            "sentiment": top_label,
            "confidence": float(confidence),
            "scores": {self.labels[i]: float(scores[i]) for i in range(len(self.labels))}
        }
    
    def predict_batch(self, texts):
        """
        Analizza il sentiment di una lista di testi.
        
        Args:
            texts (list): Lista di stringhe da analizzare
            
        Returns:
            list: Lista di dizionari con i risultati
        """
        return [self.predict(text) for text in texts]


# Entry point per test rapido
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "MLOps is amazing for continuous monitoring!",
        "This service is terrible, I want a refund.",
        "The product arrived on time."
    ]
    
    print("\n--- Test Sentiment Analysis ---")
    for text in test_texts:
        result = analyzer.predict(text)
        print(f"\nTesto: {text}")
        print(f"Risultato: {result['sentiment']} ({result['confidence']:.2%})")
