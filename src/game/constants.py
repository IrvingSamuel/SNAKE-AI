# Constantes do jogo
import pygame

# Configurações da tela
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
GAME_WIDTH = 600
GAME_HEIGHT = 600
INFO_PANEL_WIDTH = WINDOW_WIDTH - GAME_WIDTH

# Configurações do jogo
GRID_SIZE = 20
GRID_WIDTH = GAME_WIDTH // GRID_SIZE
GRID_HEIGHT = GAME_HEIGHT // GRID_SIZE

# Cores (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
LIGHT_GRAY = (192, 192, 192)

# Cores do jogo
SNAKE_COLOR = GREEN
FOOD_COLOR = RED
BACKGROUND_COLOR = BLACK
GRID_COLOR = DARK_GRAY
TEXT_COLOR = WHITE
PANEL_COLOR = (30, 30, 30)

# Velocidade do jogo
GAME_SPEED = 10
AI_SPEED = 30  # Velocidade mais rápida para treinamento

# Direções
DIRECTIONS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0)
}

# Ações da IA (0: frente, 1: direita, 2: esquerda)
AI_ACTIONS = {
    0: 'STRAIGHT',
    1: 'RIGHT',
    2: 'LEFT'
}

# Configurações da rede neural
STATE_SIZE = 28  # Tamanho expandido do estado de entrada (era 11, agora 28)
ACTION_SIZE = 3  # Número de ações possíveis
HIDDEN_SIZE = 512  # Aumentado de 256 para 512 para lidar com mais informações

# Parâmetros de treinamento
LEARNING_RATE = 0.001
GAMMA = 0.9  # Fator de desconto
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE = 100

# Configurações de fonte
pygame.font.init()
FONT_SIZE_SMALL = 16
FONT_SIZE_MEDIUM = 20
FONT_SIZE_LARGE = 24
FONT_SIZE_TITLE = 32

# Recompensas
REWARD_FOOD = 20  # Aumentado de 10 para 20 - mais incentivo para comer
REWARD_DEATH = -15  # Aumentado penalidade de -10 para -15
REWARD_MOVE = 0
REWARD_CLOSER_TO_FOOD = 2  # Aumentado de 1 para 2
REWARD_FARTHER_FROM_FOOD = -1

# Novas recompensas para estados expandidos
REWARD_AVOID_TRAP = 5
REWARD_EFFICIENT_SPACE = 3
REWARD_SIZE_BONUS = 0.5
PENALTY_HIGH_DENSITY = -3
PENALTY_TRAP_RISK = -8
PENALTY_TAIL_BLOCKING = -4
