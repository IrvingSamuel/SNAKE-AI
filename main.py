#!/usr/bin/env python3
"""
Snake AI - Jogo da Cobrinha com Rede Neural
===========================================

Um jogo da cobrinha onde uma IA aprende a jogar usando Deep Q-Learning.
Inclui interface de visualização em tempo real do processo de aprendizado.

"""

import sys
import os
import argparse
from datetime import datetime

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.helpers import (
    print_banner, setup_directories, validate_dependencies, 
    get_system_info, log_session
)

def main():
    """Função principal"""
    print_banner()
    
    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Snake AI - Jogo da Cobrinha com IA')
    parser.add_argument('--mode', choices=['play', 'train', 'train-interface'], 
                       default='play', help='Modo de execução')
    parser.add_argument('--model', type=str, help='Caminho para modelo pré-treinado')
    parser.add_argument('--games', type=int, default=1000, help='Número de jogos para treinamento')
    parser.add_argument('--speed', type=int, default=10, help='Velocidade do jogo')
    parser.add_argument('--check-deps', action='store_true', help='Verificar dependências')
    parser.add_argument('--system-info', action='store_true', help='Mostrar informações do sistema')
    
    args = parser.parse_args()
    
    # Verificar dependências se solicitado
    if args.check_deps:
        print("Verificando dependências...")
        if validate_dependencies():
            print("Todas as dependências estão OK!")
        else:
            print("Instale as dependências antes de continuar.")
            return
    
    # Mostrar informações do sistema se solicitado
    if args.system_info:
        print("Informações do Sistema:")
        info = get_system_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        return
    
    # Configurar diretórios
    setup_directories()
    
    # Verificar dependências críticas
    try:
        import pygame
        import torch
        import numpy as np
    except ImportError as e:
        print(f"Erro: Dependência faltando - {e}")
        print("Execute: pip install -r requirements.txt")
        return
    
    start_time = datetime.now()
    
    try:
        if args.mode == 'play':
            run_game_interface(args)
        elif args.mode == 'train':
            run_training(args)
        elif args.mode == 'train-interface':
            run_training_interface(args)
    
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário.")
    except Exception as e:
        print(f"Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Log da sessão
        duration = (datetime.now() - start_time).total_seconds()
        log_session(args.mode, duration)
        print("\nObrigado por usar o Snake AI!")

def run_game_interface(args):
    """Executa a interface principal do jogo"""
    print("Iniciando interface do jogo...")
    
    from src.interface.game_interface import GameInterface
    
    # Criar e executar interface
    interface = GameInterface()
    
    # Carregar modelo se especificado
    if args.model:
        if os.path.exists(args.model):
            interface.agent.load_model(args.model)
            print(f"Modelo carregado: {args.model}")
        else:
            print(f"Aviso: Modelo não encontrado: {args.model}")
    
    interface.run()

def run_training(args):
    """Executa treinamento via linha de comando"""
    print("Iniciando treinamento...")
    print(f"Número de jogos: {args.games}")
    
    from src.ai.training import train_agent
    
    # Treinar agente
    agent = train_agent(max_games=args.games)
    
    print("Treinamento concluído!")
    print(f"Record alcançado: {agent.record}")
    print(f"Jogos completados: {agent.n_games}")

def run_training_interface(args):
    """Executa interface dedicada ao treinamento"""
    print("Iniciando interface de treinamento...")
    
    from src.interface.training_interface import TrainingInterface
    
    # Criar e executar interface de treinamento
    interface = TrainingInterface()
    interface.run()

def quick_test():
    """Teste rápido do sistema"""
    print("Executando teste rápido...")
    
    try:
        # Testar imports
        from src.game.snake_game import SnakeGame
        from src.ai.agent import Agent
        
        # Testar jogo básico
        game = SnakeGame()
        agent = Agent()
        
        print("✓ Jogo inicializado")
        print("✓ Agente criado")
        
        # Testar algumas iterações
        for i in range(10):
            state = agent.get_state(game)
            action = agent.get_action(state)
            reward, done, score = game.play_step(action)
            
            if done:
                game.reset()
        
        print("✓ Teste básico passou")
        print("Sistema funcionando corretamente!")
        
    except Exception as e:
        print(f"✗ Erro no teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Se executado sem argumentos, mostrar menu interativo
    if len(sys.argv) == 1:
        print_banner()
        
        print("Escolha uma opção:")
        print("1. Jogar (Interface Completa)")
        print("2. Treinamento Rápido")
        print("3. Interface de Treinamento")
        print("4. Teste Rápido do Sistema")
        print("5. Verificar Dependências")
        print("6. Sair")
        
        try:
            choice = input("\nSua escolha (1-6): ").strip()
            
            if choice == "1":
                sys.argv = [sys.argv[0], "--mode", "play"]
            elif choice == "2":
                games = input("Número de jogos (padrão 1000): ").strip()
                if not games:
                    games = "1000"
                sys.argv = [sys.argv[0], "--mode", "train", "--games", games]
            elif choice == "3":
                sys.argv = [sys.argv[0], "--mode", "train-interface"]
            elif choice == "4":
                quick_test()
                sys.exit(0)
            elif choice == "5":
                sys.argv = [sys.argv[0], "--check-deps"]
            elif choice == "6":
                print("Até logo!")
                sys.exit(0)
            else:
                print("Opção inválida!")
                sys.exit(1)
                
        except (KeyboardInterrupt, EOFError):
            print("\nPrograma interrompido.")
            sys.exit(0)
    
    main()
