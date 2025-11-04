#!/usr/bin/env python3
"""
Debug da inicialização do jogo Snake AI
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.game.snake_game import SnakeGame, Direction
from src.game.constants import *
from src.ai.agent import Agent

def debug_initialization():
    print("=== DEBUG DA INICIALIZAÇÃO ===")
    
    # Criar jogo
    game = SnakeGame()
    print(f"Grid dimensions: {game.grid_width} x {game.grid_height}")
    print(f"Initial head position: {game.head}")
    print(f"Initial snake: {game.snake}")
    print(f"Initial direction: {game.direction}")
    print(f"Food position: {game.food}")
    
    # Verificar collision na posição inicial
    print(f"Collision at start: {game.is_collision()}")
    
    # Criar agente
    agent = Agent()
    print(f"Agent created, epsilon: {agent.epsilon}")
    
    # Testar primeiro movimento
    state = agent.get_state(game)
    print(f"Initial state: {state}")
    
    action = agent.get_action(state)
    print(f"First action: {action}")
    
    # Executar um step
    print("\n=== PRIMEIRO MOVIMENTO ===")
    reward, game_over, score = game.play_step(action)
    print(f"After first step:")
    print(f"  Head: {game.head}")
    print(f"  Snake: {game.snake}")
    print(f"  Direction: {game.direction}")
    print(f"  Game over: {game_over}")
    print(f"  Reward: {reward}")
    print(f"  Collision: {game.is_collision()}")
    
    # Verificar se a cabeça está fora dos limites
    head = game.head
    if head.x >= game.grid_width or head.x < 0:
        print(f"ERRO: Head X fora dos limites! X={head.x}, max={game.grid_width-1}")
    if head.y >= game.grid_height or head.y < 0:
        print(f"ERRO: Head Y fora dos limites! Y={head.y}, max={game.grid_height-1}")
    
    # Verificar se a cabeça está na snake
    if head in game.snake[1:]:
        print(f"ERRO: Head colidindo com corpo! Head={head}, Body={game.snake[1:]}")

if __name__ == "__main__":
    debug_initialization()
