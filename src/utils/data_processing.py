import pickle
import json
import os
import numpy as np
from datetime import datetime

class DataProcessor:
    """Classe para processar e salvar dados de treinamento"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        """Garante que o diretório de dados existe"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def save_training_data(self, agent, filename=None):
        """Salva dados de treinamento do agente"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.pkl"
        
        filepath = os.path.join(self.data_dir, filename)
        
        data = {
            'scores': agent.scores,
            'mean_scores': agent.mean_scores,
            'n_games': agent.n_games,
            'record': agent.record,
            'total_score': agent.total_score,
            'epsilon': agent.epsilon,
            'memory_size': len(agent.memory),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Dados de treinamento salvos em: {filepath}")
        return filepath
    
    def load_training_data(self, filename):
        """Carrega dados de treinamento"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Arquivo não encontrado: {filepath}")
            return None
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Dados de treinamento carregados de: {filepath}")
        return data
    
    def save_config(self, config, filename='config.json'):
        """Salva configurações em JSON"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Configuração salva em: {filepath}")
    
    def load_config(self, filename='config.json'):
        """Carrega configurações de JSON"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Arquivo de configuração não encontrado: {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        print(f"Configuração carregada de: {filepath}")
        return config
    
    def export_csv(self, agent, filename=None):
        """Exporta dados de treinamento para CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_export_{timestamp}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Preparar dados
        data = []
        for i, (score, mean_score) in enumerate(zip(agent.scores, agent.mean_scores)):
            data.append([i + 1, score, mean_score])
        
        # Salvar CSV
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Game', 'Score', 'Mean_Score'])
            writer.writerows(data)
        
        print(f"Dados exportados para CSV: {filepath}")
        return filepath

class PerformanceAnalyzer:
    """Classe para analisar performance do treinamento"""
    
    def __init__(self):
        pass
    
    def analyze_training(self, scores, window_size=100):
        """Analisa o progresso do treinamento"""
        if len(scores) < window_size:
            window_size = len(scores)
        
        recent_scores = scores[-window_size:]
        
        analysis = {
            'total_games': len(scores),
            'recent_games': window_size,
            'best_score': max(scores) if scores else 0,
            'worst_score': min(scores) if scores else 0,
            'recent_best': max(recent_scores) if recent_scores else 0,
            'recent_worst': min(recent_scores) if recent_scores else 0,
            'overall_mean': np.mean(scores) if scores else 0,
            'recent_mean': np.mean(recent_scores) if recent_scores else 0,
            'overall_std': np.std(scores) if scores else 0,
            'recent_std': np.std(recent_scores) if recent_scores else 0,
            'improvement_trend': self._calculate_trend(scores) if len(scores) > 10 else 0
        }
        
        return analysis
    
    def _calculate_trend(self, scores, window=50):
        """Calcula a tendência de melhoria"""
        if len(scores) < window * 2:
            return 0
        
        early_mean = np.mean(scores[:window])
        late_mean = np.mean(scores[-window:])
        
        return late_mean - early_mean
    
    def get_performance_report(self, agent):
        """Gera relatório completo de performance"""
        analysis = self.analyze_training(agent.scores)
        
        report = f"""
=== RELATÓRIO DE PERFORMANCE - SNAKE AI ===
Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

ESTATÍSTICAS GERAIS:
- Total de Jogos: {analysis['total_games']}
- Melhor Score: {analysis['best_score']}
- Pior Score: {analysis['worst_score']}
- Média Geral: {analysis['overall_mean']:.2f}
- Desvio Padrão: {analysis['overall_std']:.2f}

ÚLTIMOS {analysis['recent_games']} JOGOS:
- Melhor Score: {analysis['recent_best']}
- Pior Score: {analysis['recent_worst']}
- Média Recente: {analysis['recent_mean']:.2f}
- Desvio Padrão: {analysis['recent_std']:.2f}

TENDÊNCIA:
- Melhoria: {analysis['improvement_trend']:.2f} pontos
- Status: {"Melhorando" if analysis['improvement_trend'] > 0 else "Estagnado/Piorando"}

PARÂMETROS ATUAIS:
- Epsilon: {agent.epsilon:.4f}
- Record Atual: {agent.record}
- Memória Utilizada: {len(agent.memory)}/{agent.memory.maxlen}
"""
        
        return report
