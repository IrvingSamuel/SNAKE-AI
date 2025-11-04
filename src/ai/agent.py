import torch
import random
import numpy as np
from collections import deque
from ..game.snake_game import SnakeGame, Direction, Point
from ..game.game_state import GameState
from ..game.constants import *
from .neural_network import LinearQNet, DQN, QTrainer

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START  # randomness
        self.gamma = GAMMA  # discount rate
        self.memory = deque(maxlen=MEMORY_SIZE)  # popleft()
        # Usar DQN mais robusta para o estado expandido
        self.model = DQN(STATE_SIZE, HIDDEN_SIZE, ACTION_SIZE)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)
        
        # Para estatísticas
        self.scores = []
        self.mean_scores = []
        self.total_score = 0
        self.record = 0

    def get_state(self, game):
        game_state = GameState(game)
        return game_state.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft se MAX_MEMORY for atingido

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # lista de tuplas
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # movimentos aleatórios: tradeoff exploration / exploitation
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.n_games / 1000)
        final_move = [0, 0, 0]
        
        if random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def load_model(self, file_path):
        """Carrega um modelo pré-treinado (compatível com LinearQNet e DQN)"""
        try:
            import os
            if not os.path.exists(file_path):
                print(f"Arquivo de modelo não encontrado: {file_path}")
                return False
                
            checkpoint = torch.load(file_path, weights_only=False)
            
            # Verificar se é um modelo antigo (LinearQNet) ou novo (DQN)
            is_old_model = False
            
            # Se for apenas o state_dict
            if isinstance(checkpoint, dict):
                # Verificar se tem keys do modelo antigo
                if 'linear1.weight' in checkpoint or ('model_state_dict' in checkpoint and 'linear1.weight' in checkpoint['model_state_dict']):
                    is_old_model = True
                    print("Detectado modelo antigo (LinearQNet). Convertendo para DQN...")
                
                # Modelo novo (DQN)
                if 'fc1.weight' in checkpoint:
                    self.model.load_state_dict(checkpoint)
                    print(f"Modelo DQN carregado de {file_path}")
                    self.model.eval()
                    return True
                
                # Checkpoint completo novo
                elif 'model_state_dict' in checkpoint and 'fc1.weight' in checkpoint['model_state_dict']:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.n_games = checkpoint.get('n_games', 0)
                    self.epsilon = checkpoint.get('epsilon', EPSILON_END)
                    self.record = checkpoint.get('record', 0)
                    self.scores = checkpoint.get('scores', [])
                    self.mean_scores = checkpoint.get('mean_scores', [])
                    self.total_score = checkpoint.get('total_score', 0)
                    print(f"Checkpoint DQN carregado de {file_path}")
                    print(f"  Jogos: {self.n_games}, Record: {self.record}, Epsilon: {self.epsilon:.3f}")
                    self.model.eval()
                    return True
                
                # Modelo antigo ou checkpoint antigo
                elif is_old_model:
                    # Criar modelo antigo temporário para carregar os pesos
                    old_model = LinearQNet(11, 256, ACTION_SIZE)  # Configuração antiga
                    
                    if 'model_state_dict' in checkpoint:
                        # Checkpoint completo antigo
                        old_model.load_state_dict(checkpoint['model_state_dict'])
                        self.n_games = checkpoint.get('n_games', 0)
                        self.epsilon = checkpoint.get('epsilon', EPSILON_END)
                        self.record = checkpoint.get('record', 0)
                        self.scores = checkpoint.get('scores', [])
                        self.mean_scores = checkpoint.get('mean_scores', [])
                        self.total_score = checkpoint.get('total_score', 0)
                    else:
                        # State dict antigo
                        old_model.load_state_dict(checkpoint)
                    
                    print(f"Modelo antigo carregado. Iniciando novo treinamento com DQN expandida.")
                    print(f"  Jogos anteriores: {self.n_games}, Record anterior: {self.record}")
                    return True
                
                else:
                    print(f"Formato de arquivo não reconhecido: {file_path}")
                    return False
            else:
                print(f"Formato de checkpoint inválido: {file_path}")
                return False
                
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            print("Iniciando novo treinamento...")
            return False
    
    def save_model(self, file_path=None, save_training_data=True):
        """Salva o modelo atual"""
        import os
        
        if file_path is None:
            file_path = f'models/model_game_{self.n_games}.pth'
        
        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            if save_training_data:
                # Salvar checkpoint completo
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'n_games': self.n_games,
                    'epsilon': self.epsilon,
                    'record': self.record,
                    'scores': self.scores,
                    'mean_scores': self.mean_scores,
                    'total_score': self.total_score,
                    'gamma': self.gamma,
                    'learning_rate': LEARNING_RATE
                }
                torch.save(checkpoint, file_path)
                print(f"Checkpoint completo salvo em {file_path}")
            else:
                # Salvar apenas o modelo
                torch.save(self.model.state_dict(), file_path)
                print(f"Modelo salvo em {file_path}")
            return True
        except Exception as e:
            print(f"Erro ao salvar modelo: {e}")
            return False
    
    def find_latest_model(self):
        """Encontra o modelo mais recente"""
        import os
        import glob
        
        model_patterns = [
            'models/checkpoint_*.pth',
            'models/model_game_*.pth',
            'models/auto_save_*.pth',
            'models/model_final.pth'
        ]
        
        latest_file = None
        latest_games = -1
        
        for pattern in model_patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    if 'final' in file:
                        # Verificar se o arquivo final tem dados válidos
                        checkpoint = torch.load(file, weights_only=False)
                        if isinstance(checkpoint, dict) and 'n_games' in checkpoint:
                            games = checkpoint['n_games']
                            if games > latest_games:
                                latest_games = games
                                latest_file = file
                    else:
                        # Extrair número de jogos do nome do arquivo
                        import re
                        match = re.search(r'(\d+)', os.path.basename(file))
                        if match:
                            games = int(match.group(1))
                            if games > latest_games:
                                latest_games = games
                                latest_file = file
                except:
                    continue
        
        return latest_file, latest_games
    
    def auto_load_latest(self):
        """Carrega automaticamente o modelo mais recente"""
        latest_file, games = self.find_latest_model()
        if latest_file:
            print(f"Modelo mais recente encontrado: {latest_file} ({games} jogos)")
            return self.load_model(latest_file)
        else:
            print("Nenhum modelo anterior encontrado. Iniciando treinamento do zero.")
            return False
    
    def get_stats(self):
        """Retorna estatísticas do treinamento"""
        return {
            'games_played': self.n_games,
            'epsilon': self.epsilon,
            'record_score': self.record,
            'mean_score': np.mean(self.scores[-100:]) if self.scores else 0,
            'recent_scores': self.scores[-10:] if len(self.scores) >= 10 else self.scores,
            'total_score': self.total_score
        }
