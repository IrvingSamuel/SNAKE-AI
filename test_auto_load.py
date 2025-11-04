#!/usr/bin/env python3
"""
Teste do sistema de carregamento automático de modelos
"""

import sys
import os

# Adicionar src ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

# Importações com path correto
from src.ai.agent import Agent

def test_auto_load():
    """Testa o carregamento automático de modelos"""
    print("=== Teste do Sistema de Carregamento Automático ===")
    
    # Criar agente - deve carregar automaticamente o modelo mais recente
    print("\n1. Criando agente (deve carregar modelo automaticamente)...")
    agent = Agent()
    
    # Tentar carregar o modelo mais recente
    print("   Tentando carregar modelo mais recente...")
    agent.auto_load_latest()
    
    print(f"   Games jogados: {agent.n_games}")
    print(f"   Epsilon atual: {agent.epsilon}")
    print(f"   Record atual: {agent.record}")
    
    # Verificar se realmente carregou um modelo
    if agent.n_games > 0:
        print(f"✅ SUCESSO: Modelo carregado automaticamente!")
        print(f"   Continuing training from game {agent.n_games}")
        print(f"   Current exploration rate: {agent.epsilon:.4f}")
        print(f"   Best score achieved: {agent.record}")
    else:
        print("❌ FALHA: Nenhum modelo foi carregado automaticamente")
        print("   Starting from scratch")
    
    print("\n2. Testando salvamento de checkpoint...")
    # Simular algum progresso e salvar
    agent.n_games += 1
    agent.memory.append(([0,0,0,0,0,0,0,0,0,0,0], 0, 10, [0,0,0,0,0,0,0,0,0,0,0], False))
    agent.save_model()
    print(f"✅ Checkpoint salvo para game {agent.n_games}")
    
    print("\n=== Teste Concluído ===")

if __name__ == "__main__":
    test_auto_load()
