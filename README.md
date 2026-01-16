# 🚀 MachineInnovators Inc. - Sentiment Analysis System

[![CI/CD Pipeline](https://github.com/cellli/Machine-Innovators-Sentiment/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/cellli/Machine-Innovators-Sentiment/actions/workflows/ci_pipeline.yml)

Sistema MLOps completo per il monitoraggio della reputazione online attraverso l'analisi del sentiment.

## 📋 Panoramica

Questo progetto implementa un sistema end-to-end per:
- **Analisi automatica del sentiment** di testi da social media
- **Monitoraggio continuo** delle performance del modello
- **Pipeline CI/CD** per testing e deployment automatizzati
- **Sistema di retraining** per mantenere il modello aggiornato

## 🏗️ Architettura

```
machine_innovators_sentiment/
├── app.py              # Modulo principale (SentimentAnalyzer)
├── test_app.py         # Test unitari
├── monitoring.py       # Sistema di monitoraggio
├── retrain.py          # Logica di retraining
├── requirements.txt    # Dipendenze Python
└── .github/
    └── workflows/
        └── ci_pipeline.yml  # Pipeline CI/CD
```

## 🤖 Modello

- **Modello**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Architettura**: RoBERTa (Transformer)
- **Task**: Classificazione a 3 classi (Negative, Neutral, Positive)
- **Training Data**: TweetEval dataset

### Performance sul Test Set

| Metrica | Valore |
|---------|--------|
| Accuracy | 72% |
| F1-Score (macro) | 70% |
| Confidenza media | 85% |

## 🚀 Quick Start

```python
from app import SentimentAnalyzer

# Inizializza l'analizzatore
analyzer = SentimentAnalyzer()

# Analizza un testo
result = analyzer.predict("I love this product!")
print(result)
# Output: {'sentiment': 'Positive', 'confidence': 0.95, 'scores': {...}}
```

## 📊 Sistema di Monitoraggio

Il modulo `monitoring.py` traccia:
- Distribuzione dei sentiment nel tempo
- Confidenza media delle predizioni
- Drift detection per trigger di retraining

```python
from monitoring import SentimentMonitor

monitor = SentimentMonitor()
alerts = monitor.log_prediction(text, result)
print(monitor.get_summary_report())
```

## 🔄 CI/CD Pipeline

La pipeline GitHub Actions esegue automaticamente:
1. **Build & Test**: Test unitari ad ogni push
2. **Performance Check**: Benchmark su main branch
3. **Scheduled Retraining**: Check settimanale per retraining

## ⚠️ Limitazioni Note

- Il modello è ottimizzato per **testi in inglese**
- Testi in italiano possono avere accuratezza ridotta
- Per supporto multilingua, considerare modelli come `xlm-roberta`

## 📝 Licenza

MIT License - MachineInnovators Inc. 2024
