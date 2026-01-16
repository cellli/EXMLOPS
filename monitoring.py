"""
MachineInnovators Inc. - Monitoring Module
==========================================
Sistema di monitoraggio continuo per il modello di Sentiment Analysis.

Funzionalità:
- Logging delle predizioni con timestamp
- Calcolo di metriche in tempo reale
- Rilevamento del drift delle performance
- Trigger per il retraining automatico
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque


class SentimentMonitor:
    """
    Sistema di monitoraggio per il modello di Sentiment Analysis.
    
    Traccia:
    - Distribuzione dei sentiment nel tempo
    - Confidenza media delle predizioni
    - Volume di richieste
    - Drift detection per trigger di retraining
    """
    
    def __init__(self, log_file="monitoring_logs.csv", window_size=100):
        """
        Inizializza il sistema di monitoraggio.
        
        Args:
            log_file (str): Path del file CSV per i log
            window_size (int): Dimensione della finestra per le metriche rolling
        """
        self.log_file = log_file
        self.window_size = window_size
        
        # Buffer per metriche rolling
        self.confidence_buffer = deque(maxlen=window_size)
        self.sentiment_buffer = deque(maxlen=window_size)
        
        # Soglie per alert
        self.confidence_threshold = 0.6  # Alert se confidenza media < 60%
        self.drift_threshold = 0.2  # Alert se distribuzione cambia > 20%
        
        # Baseline della distribuzione (da calibrare sul training set)
        self.baseline_distribution = {
            'Negative': 0.33,
            'Neutral': 0.34,
            'Positive': 0.33
        }
        
        # Inizializza il file di log se non esiste
        if not os.path.exists(log_file):
            df_init = pd.DataFrame(columns=[
                "timestamp", "input_text", "sentiment", 
                "confidence", "neg_score", "neu_score", "pos_score"
            ])
            df_init.to_csv(log_file, index=False)
            print(f"📁 File di log creato: {log_file}")
    
    def log_prediction(self, text, result):
        """
        Registra una predizione nel sistema di monitoraggio.
        
        Args:
            text (str): Testo analizzato
            result (dict): Risultato della predizione
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Aggiorna i buffer
        self.confidence_buffer.append(result['confidence'])
        self.sentiment_buffer.append(result['sentiment'])
        
        # Salva nel CSV
        new_data = pd.DataFrame([[
            timestamp,
            text[:200],  # Tronca testi lunghi
            result['sentiment'],
            result['confidence'],
            result['scores']['Negative'],
            result['scores']['Neutral'],
            result['scores']['Positive']
        ]], columns=[
            "timestamp", "input_text", "sentiment", 
            "confidence", "neg_score", "neu_score", "pos_score"
        ])
        new_data.to_csv(self.log_file, mode='a', header=False, index=False)
        
        # Controlla alert
        alerts = self.check_alerts()
        return alerts
    
    def get_current_metrics(self):
        """
        Calcola le metriche correnti sulla finestra rolling.
        
        Returns:
            dict: Metriche correnti del sistema
        """
        if len(self.confidence_buffer) == 0:
            return {"status": "no_data", "message": "Nessun dato disponibile"}
        
        # Calcola distribuzione corrente
        sentiment_counts = {}
        for s in ['Negative', 'Neutral', 'Positive']:
            sentiment_counts[s] = self.sentiment_buffer.count(s) / len(self.sentiment_buffer)
        
        return {
            "status": "ok",
            "total_predictions": len(self.confidence_buffer),
            "avg_confidence": np.mean(self.confidence_buffer),
            "min_confidence": np.min(self.confidence_buffer),
            "max_confidence": np.max(self.confidence_buffer),
            "sentiment_distribution": sentiment_counts,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def check_alerts(self):
        """
        Controlla se ci sono condizioni di alert.
        
        Returns:
            list: Lista di alert attivi
        """
        alerts = []
        
        if len(self.confidence_buffer) < 10:
            return alerts  # Non abbastanza dati
        
        # Check 1: Confidenza media troppo bassa
        avg_conf = np.mean(self.confidence_buffer)
        if avg_conf < self.confidence_threshold:
            alerts.append({
                "type": "LOW_CONFIDENCE",
                "severity": "WARNING",
                "message": f"Confidenza media ({avg_conf:.2%}) sotto la soglia ({self.confidence_threshold:.2%})",
                "action": "Considerare retraining del modello"
            })
        
        # Check 2: Drift nella distribuzione
        current_dist = {}
        for s in ['Negative', 'Neutral', 'Positive']:
            current_dist[s] = self.sentiment_buffer.count(s) / len(self.sentiment_buffer)
        
        max_drift = max(
            abs(current_dist[s] - self.baseline_distribution[s])
            for s in ['Negative', 'Neutral', 'Positive']
        )
        
        if max_drift > self.drift_threshold:
            alerts.append({
                "type": "DISTRIBUTION_DRIFT",
                "severity": "INFO",
                "message": f"Drift rilevato nella distribuzione ({max_drift:.2%})",
                "current_distribution": current_dist,
                "action": "Verificare se i dati in input sono cambiati"
            })
        
        return alerts
    
    def should_retrain(self):
        """
        Determina se è necessario un retraining del modello.
        
        Returns:
            tuple: (bool, str) - (necessità di retraining, motivazione)
        """
        if len(self.confidence_buffer) < self.window_size:
            return False, "Dati insufficienti per la valutazione"
        
        avg_conf = np.mean(self.confidence_buffer)
        
        if avg_conf < 0.5:
            return True, f"Confidenza critica: {avg_conf:.2%}"
        
        # Controlla trend negativo della confidenza
        recent = list(self.confidence_buffer)[-20:]
        older = list(self.confidence_buffer)[:20]
        
        if np.mean(recent) < np.mean(older) - 0.1:
            return True, "Trend negativo della confidenza rilevato"
        
        return False, "Performance nella norma"
    
    def get_summary_report(self):
        """
        Genera un report riassuntivo delle performance.
        
        Returns:
            str: Report formattato
        """
        metrics = self.get_current_metrics()
        alerts = self.check_alerts()
        retrain_needed, retrain_reason = self.should_retrain()
        
        report = [
            "\n" + "="*60,
            "📊 MONITORING REPORT - MachineInnovators Inc.",
            "="*60,
            f"\n⏰ Timestamp: {metrics.get('timestamp', 'N/A')}",
            f"📈 Predizioni totali (finestra): {metrics.get('total_predictions', 0)}",
            f"\n🎯 PERFORMANCE METRICHE:",
            f"   • Confidenza media: {metrics.get('avg_confidence', 0):.2%}",
            f"   • Confidenza min/max: {metrics.get('min_confidence', 0):.2%} / {metrics.get('max_confidence', 0):.2%}",
            f"\n📊 DISTRIBUZIONE SENTIMENT:"
        ]
        
        if 'sentiment_distribution' in metrics:
            for sent, pct in metrics['sentiment_distribution'].items():
                report.append(f"   • {sent}: {pct:.1%}")
        
        report.append(f"\n⚠️ ALERT ATTIVI: {len(alerts)}")
        for alert in alerts:
            report.append(f"   [{alert['severity']}] {alert['message']}")
        
        report.append(f"\n🔄 RETRAINING: {'NECESSARIO' if retrain_needed else 'Non necessario'}")
        report.append(f"   Motivo: {retrain_reason}")
        report.append("\n" + "="*60)
        
        return "\n".join(report)


# Entry point per test
if __name__ == "__main__":
    monitor = SentimentMonitor(log_file="test_monitoring.csv")
    
    # Simula alcune predizioni
    test_results = [
        {"sentiment": "Positive", "confidence": 0.95, "scores": {"Negative": 0.02, "Neutral": 0.03, "Positive": 0.95}},
        {"sentiment": "Negative", "confidence": 0.88, "scores": {"Negative": 0.88, "Neutral": 0.08, "Positive": 0.04}},
        {"sentiment": "Neutral", "confidence": 0.65, "scores": {"Negative": 0.15, "Neutral": 0.65, "Positive": 0.20}},
    ]
    
    for i, result in enumerate(test_results):
        alerts = monitor.log_prediction(f"Test text {i}", result)
        if alerts:
            print(f"Alert: {alerts}")
    
    print(monitor.get_summary_report())
