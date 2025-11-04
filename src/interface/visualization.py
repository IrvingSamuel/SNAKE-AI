import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import threading
import time

class TrainingVisualizer:
    def __init__(self, max_points=1000):
        self.max_points = max_points
        self.scores = deque(maxlen=max_points)
        self.mean_scores = deque(maxlen=max_points)
        self.epsilons = deque(maxlen=max_points)
        self.games = deque(maxlen=max_points)
        
        # Configurar matplotlib para modo não-bloqueante
        plt.ion()
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Snake AI - Progresso do Treinamento', fontsize=16)
        
        # Configurar os subplots
        self.setup_plots()
        
        # Para animação em tempo real
        self.running = False
        
    def setup_plots(self):
        """Configura os gráficos"""
        # Gráfico 1: Scores por jogo
        self.ax1.set_title('Scores por Jogo')
        self.ax1.set_xlabel('Jogo')
        self.ax1.set_ylabel('Score')
        self.ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Média móvel dos scores
        self.ax2.set_title('Média Móvel dos Scores')
        self.ax2.set_xlabel('Jogo')
        self.ax2.set_ylabel('Score Médio')
        self.ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Epsilon (taxa de exploração)
        self.ax3.set_title('Taxa de Exploração (Epsilon)')
        self.ax3.set_xlabel('Jogo')
        self.ax3.set_ylabel('Epsilon')
        self.ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Distribuição dos scores recentes
        self.ax4.set_title('Distribuição dos Últimos 100 Scores')
        self.ax4.set_xlabel('Score')
        self.ax4.set_ylabel('Frequência')
        self.ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def update_data(self, score, mean_score, epsilon, game_number):
        """Atualiza os dados para visualização"""
        self.scores.append(score)
        self.mean_scores.append(mean_score)
        self.epsilons.append(epsilon)
        self.games.append(game_number)
        
    def update_plots(self):
        """Atualiza todos os gráficos"""
        if len(self.scores) == 0:
            return
            
        # Limpar plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Reconfigurar
        self.setup_plots()
        
        games_list = list(self.games)
        scores_list = list(self.scores)
        mean_scores_list = list(self.mean_scores)
        epsilons_list = list(self.epsilons)
        
        # Plot 1: Scores individuais
        self.ax1.plot(games_list, scores_list, 'b-', alpha=0.6, linewidth=1)
        if len(scores_list) > 0:
            self.ax1.set_ylim(0, max(scores_list) * 1.1)
        
        # Plot 2: Média móvel
        self.ax2.plot(games_list, mean_scores_list, 'r-', linewidth=2)
        if len(mean_scores_list) > 0:
            self.ax2.set_ylim(0, max(mean_scores_list) * 1.1)
        
        # Plot 3: Epsilon
        self.ax3.plot(games_list, epsilons_list, 'g-', linewidth=2)
        self.ax3.set_ylim(0, 1)
        
        # Plot 4: Histograma dos últimos scores
        recent_scores = list(self.scores)[-100:] if len(self.scores) >= 100 else list(self.scores)
        if recent_scores:
            self.ax4.hist(recent_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        
        # Adicionar estatísticas
        if scores_list:
            stats_text = f'Último: {scores_list[-1]}\nMáximo: {max(scores_list)}\nMédia: {mean_scores_list[-1]:.1f}'
            self.ax1.text(0.02, 0.98, stats_text, transform=self.ax1.transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        
    def start_realtime_update(self, update_interval=1.0):
        """Inicia atualização em tempo real"""
        self.running = True
        
        def update_loop():
            while self.running:
                self.update_plots()
                time.sleep(update_interval)
        
        self.update_thread = threading.Thread(target=update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def stop_realtime_update(self):
        """Para a atualização em tempo real"""
        self.running = False
        
    def save_plots(self, filename='training_progress.png'):
        """Salva os gráficos em arquivo"""
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gráficos salvos em {filename}")

class RealTimeMonitor:
    """Monitor em tempo real para acompanhar o treinamento"""
    def __init__(self, agent):
        self.agent = agent
        self.visualizer = TrainingVisualizer()
        self.last_game_count = 0
        
    def update(self):
        """Atualiza o monitor com dados mais recentes"""
        current_games = self.agent.n_games
        
        if current_games > self.last_game_count:
            # Novos jogos foram completados
            stats = self.agent.get_stats()
            
            if stats['recent_scores']:
                latest_score = stats['recent_scores'][-1]
                mean_score = stats['mean_score']
                epsilon = stats['epsilon']
                
                self.visualizer.update_data(latest_score, mean_score, epsilon, current_games)
                
            self.last_game_count = current_games
    
    def start_monitoring(self):
        """Inicia o monitoramento"""
        self.visualizer.start_realtime_update()
        print("Monitor de treinamento iniciado!")
        
    def stop_monitoring(self):
        """Para o monitoramento"""
        self.visualizer.stop_realtime_update()
        
    def save_report(self, filename='training_report.png'):
        """Salva um relatório visual do treinamento"""
        self.visualizer.save_plots(filename)
