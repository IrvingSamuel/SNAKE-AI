import matplotlib.pyplot as plt
import numpy as np
from .agent import Agent
from ..game.snake_game import SnakeGame
from ..game.game_state import GameState
from ..game.constants import *

class Trainer:
    def __init__(self):
        self.agent = Agent()
        self.game = SnakeGame()
        self.game_state = GameState(self.game)
        
        # Para plotar gráficos
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    def plot_training_progress(self, scores, mean_scores):
        """Plota o progresso do treinamento"""
        plt.clf()  # Clear the current figure
        
        # Gráfico de scores
        self.ax1.clear()
        self.ax1.set_title('Training Progress - Scores')
        self.ax1.set_xlabel('Number of Games')
        self.ax1.set_ylabel('Score')
        self.ax1.plot(scores, label='Score', alpha=0.7)
        self.ax1.plot(mean_scores, label='Mean Score', color='red', linewidth=2)
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Gráfico de epsilon (exploration rate)
        self.ax2.clear()
        self.ax2.set_title('Epsilon Decay')
        self.ax2.set_xlabel('Number of Games')
        self.ax2.set_ylabel('Epsilon')
        epsilons = [EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * i / 1000) 
                   for i in range(len(scores))]
        self.ax2.plot(epsilons, color='green', linewidth=2)
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def train(self, max_games=1000, save_every=100, plot_every=10):
        """Função principal de treinamento"""
        print("Iniciando treinamento...")
        print(f"Máximo de jogos: {max_games}")
        print(f"Salvando modelo a cada {save_every} jogos")
        
        # Tentar carregar modelo anterior
        if self.agent.n_games == 0:
            self.agent.auto_load_latest()
        
        start_games = self.agent.n_games
        target_games = start_games + max_games
        
        print(f"Continuando do jogo {start_games} até {target_games}")
        
        while self.agent.n_games < target_games:
            # Estado atual
            state_old = self.agent.get_state(self.game)
            distance_old = self.game_state.get_distance_to_food()

            # Obter movimento
            final_move = self.agent.get_action(state_old)

            # Executar movimento e obter novo estado
            reward, done, score = self.game.play_step(final_move)
            state_new = self.agent.get_state(self.game)
            
            # Sistema de recompensas expandido
            if not done:
                # Recompensa baseada na distância
                distance_new = self.game_state.get_distance_to_food()
                if distance_new < distance_old:
                    reward += REWARD_CLOSER_TO_FOOD
                elif distance_new > distance_old:
                    reward += REWARD_FARTHER_FROM_FOOD
                
                # Recompensas adicionais baseadas no estado expandido
                game_info = self.game_state.get_game_info()
                
                # Penalizar alta densidade corporal (evitar armadilhas)
                if game_info['body_density'] > 0.7:
                    reward += PENALTY_HIGH_DENSITY
                
                # Recompensar boa eficiência de espaço livre
                if game_info['free_space_ratio'] > 0.5:
                    reward += REWARD_EFFICIENT_SPACE
                
                # Penalizar fortemente detecção de armadilhas
                if game_info['trap_risk']:
                    reward += PENALTY_TRAP_RISK
                
                # Penalizar bloqueio da cauda
                if game_info['tail_blocking']:
                    reward += PENALTY_TAIL_BLOCKING
                
                # Recompensa proporcional ao tamanho da cobra (incentiva crescimento)
                size_bonus = len(self.game.snake) * REWARD_SIZE_BONUS
                reward += size_bonus

            # Treinar memória curta
            self.agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # Lembrar
            self.agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # Treinar memória longa, plotar resultado
                self.game.reset()
                self.agent.n_games += 1
                self.agent.train_long_memory()

                if score > self.agent.record:
                    self.agent.record = score

                print(f'Jogo {self.agent.n_games}, Score: {score}, Record: {self.agent.record}, Epsilon: {self.agent.epsilon:.3f}')

                self.agent.scores.append(score)
                self.agent.total_score += score
                mean_score = self.agent.total_score / self.agent.n_games
                self.agent.mean_scores.append(mean_score)
                
                # Plotar a cada X jogos
                if self.agent.n_games % plot_every == 0:
                    self.plot_training_progress(self.agent.scores, self.agent.mean_scores)
                
                # Salvar modelo a cada X jogos
                if self.agent.n_games % save_every == 0:
                    self.agent.save_model(f'models/model_checkpoint_{self.agent.n_games}.pth')
        
        # Salvar modelo final
        self.agent.save_model('models/model_final.pth')
        print(f"Treinamento concluído! Modelo salvo. Record: {self.agent.record}")
        
        return self.agent

def train_agent(max_games=1000):
    """Função conveniente para iniciar o treinamento"""
    trainer = Trainer()
    return trainer.train(max_games=max_games)
