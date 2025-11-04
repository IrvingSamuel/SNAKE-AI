#!/usr/bin/env python3
"""
Teste completo dos movimentos da IA
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.game.snake_game import SnakeGame, Direction
from src.ai.agent import Agent

def test_ai_movements():
    print("=== TESTE COMPLETO DOS MOVIMENTOS DA IA ===")
    
    game = SnakeGame()
    agent = Agent()
    
    print(f"Posição inicial: {game.head}")
    print(f"Direção inicial: {game.direction}")
    
    # Testar 10 movimentos
    for i in range(10):
        state = agent.get_state(game)
        action = agent.get_action(state)
        
        print(f"\nMovimento {i+1}:")
        print(f"  Estado: {state}")
        print(f"  Ação: {action}")
        print(f"  Posição antes: {game.head}")
        print(f"  Direção antes: {game.direction}")
        
        reward, game_over, score = game.play_step(action)
        
        print(f"  Posição depois: {game.head}")
        print(f"  Direção depois: {game.direction}")
        print(f"  Game over: {game_over}")
        print(f"  Reward: {reward}")
        
        if game_over:
            print(f"  GAME OVER! Motivo: {'Colisão' if game.is_collision() else 'Timeout'}")
            break
    
    print(f"\nFinal: Score = {game.score}, Jogos = {agent.n_games}")

if __name__ == "__main__":
    test_ai_movements()
