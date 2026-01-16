"""
MachineInnovators Inc. - Unit Tests
====================================
Test automatizzati per il modulo di Sentiment Analysis.
Eseguiti automaticamente dalla pipeline CI/CD.
"""

import unittest
from app import SentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):
    """
    Suite di test per la classe SentimentAnalyzer.
    """

    @classmethod
    def setUpClass(cls):
        """Inizializza il modello una volta sola per tutti i test."""
        print("\n🔧 Setup: Caricamento modello per i test...")
        cls.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        """Test: Una frase positiva deve essere classificata come Positive."""
        text = "I absolutely love this new feature, it is fantastic!"
        result = self.analyzer.predict(text)
        self.assertEqual(result['sentiment'], 'Positive')
        self.assertGreater(result['confidence'], 0.5)

    def test_negative_sentiment(self):
        """Test: Una frase negativa deve essere classificata come Negative."""
        text = "This is the worst experience I have ever had. Terrible."
        result = self.analyzer.predict(text)
        self.assertEqual(result['sentiment'], 'Negative')
        self.assertGreater(result['confidence'], 0.5)

    def test_output_structure(self):
        """Test: L'output deve contenere le chiavi corrette."""
        text = "Test message"
        result = self.analyzer.predict(text)
        
        self.assertIn('sentiment', result)
        self.assertIn('confidence', result)
        self.assertIn('scores', result)
        
        # Verifica che scores contenga tutte e 3 le classi
        self.assertIn('Negative', result['scores'])
        self.assertIn('Neutral', result['scores'])
        self.assertIn('Positive', result['scores'])

    def test_confidence_range(self):
        """Test: La confidence deve essere tra 0 e 1."""
        text = "Testing confidence range"
        result = self.analyzer.predict(text)
        
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_scores_sum_to_one(self):
        """Test: La somma degli scores deve essere circa 1."""
        text = "Testing probability distribution"
        result = self.analyzer.predict(text)
        
        total = sum(result['scores'].values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_batch_prediction(self):
        """Test: La predizione batch deve funzionare correttamente."""
        texts = ["Great!", "Terrible!", "Okay"]
        results = self.analyzer.predict_batch(texts)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('sentiment', result)


if __name__ == '__main__':
    # Esecuzione con output verboso
    unittest.main(verbosity=2)
