#!/usr/bin/env python3
"""
Teste simples dos controles do Snake AI
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pygame
from src.game.snake_game import SnakeGame, Direction
from src.game.constants import *

def test_controls():
    """Teste básico dos controles"""
    pygame.init()
    screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
    pygame.display.set_caption("Snake AI - Teste de Controles")
    clock = pygame.time.Clock()
    
    game = SnakeGame()
    game.set_display(screen)
    
    print("Teste de Controles do Snake AI")
    print("Use as setas do teclado para mover a cobrinha")
    print("Pressione ESC para sair")
    
    running = True
    while running:
        # Processar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT and game.direction != Direction.RIGHT:
                    game.direction = Direction.LEFT
                    print("Direção: ESQUERDA")
                elif event.key == pygame.K_RIGHT and game.direction != Direction.LEFT:
                    game.direction = Direction.RIGHT
                    print("Direção: DIREITA")
                elif event.key == pygame.K_UP and game.direction != Direction.DOWN:
                    game.direction = Direction.UP
                    print("Direção: CIMA")
                elif event.key == pygame.K_DOWN and game.direction != Direction.UP:
                    game.direction = Direction.DOWN
                    print("Direção: BAIXO")
        
        # Atualizar jogo
        reward, game_over, score = game.play_step()
        
        if game_over:
            print(f"Game Over! Score: {score}")
            game.reset()
        
        # Atualizar tela
        clock.tick(GAME_SPEED)
    
    pygame.quit()
    print("Teste finalizado!")

if __name__ == "__main__":
    test_controls()
