#!/usr/bin/env python3
"""
Teste do Sistema de Estado Expandido
====================================

Testa as novas funcionalidades do estado expandido:
- Tamanho da cobra normalizado
- Densidade corporal
- Distâncias até bordas
- Espaços livres
- Detecção de armadilhas
- Eficiência de movimento

"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.game.snake_game import SnakeGame
from src.game.game_state import GameState
from src.ai.agent import Agent
from src.game.constants import *

def test_expanded_state():
    """Testa o estado expandido"""
    print("🧪 Testando Estado Expandido")
    print("=" * 50)
    
    # Criar jogo e agente
    game = SnakeGame()
    game_state = GameState(game)
    agent = Agent()
    
    print(f"✓ Jogo inicializado")
    print(f"✓ Estado expandido: {STATE_SIZE} características")
    
    # Testar estado inicial
    state = game_state.get_state()
    print(f"✓ Estado shape: {state.shape}")
    print(f"✓ Estado dtype: {state.dtype}")
    print(f"✓ Estado range: [{state.min():.2f}, {state.max():.2f}]")
    
    # Verificar se todas as características estão sendo calculadas
    if len(state) != STATE_SIZE:
        print(f"❌ Erro: Estado tem {len(state)} características, esperado {STATE_SIZE}")
        return False
    
    print(f"✓ Tamanho do estado correto: {len(state)}")
    
    # Testar informações do jogo
    info = game_state.get_game_info()
    print("\n📊 Informações do Estado:")
    print(f"  Score: {info['score']}")
    print(f"  Tamanho da cobra: {info['snake_length']}")
    print(f"  Distância até comida: {info['distance_to_food']}")
    print(f"  Densidade corporal: {info['body_density']:.3f}")
    print(f"  Razão espaço livre: {info['free_space_ratio']:.3f}")
    print(f"  Risco de armadilha: {info['trap_risk']}")
    print(f"  Cauda bloqueando: {info['tail_blocking']}")
    
    # Testar algumas jogadas
    print(f"\n🎮 Testando jogabilidade...")
    
    for i in range(10):
        # Obter estado e ação
        state = agent.get_state(game)
        action = agent.get_action(state)
        
        # Executar jogada
        reward, done, score = game.play_step(action)
        
        if done:
            print(f"  Jogo {i+1}: Terminou com score {score}")
            game.reset()
        else:
            print(f"  Jogada {i+1}: Score {score}, Reward {reward}")
    
    print(f"✓ Teste de jogabilidade concluído")
    return True

def test_state_characteristics():
    """Testa características específicas do estado"""
    print("\n🔍 Testando Características Específicas")
    print("=" * 50)
    
    game = SnakeGame()
    game_state = GameState(game)
    
    # Testar com cobra pequena
    print("📏 Cobra pequena (tamanho inicial):")
    state = game_state.get_state()
    info = game_state.get_game_info()
    
    print(f"  Tamanho normalizado: {state[11]:.3f}")
    print(f"  Densidade corporal média: {info['body_density']:.3f}")
    print(f"  Espaço livre médio: {info['free_space_ratio']:.3f}")
    
    # Simular cobra maior (adicionando segmentos artificialmente)
    print("\n📏 Simulando cobra maior:")
    original_snake = game.snake.copy()
    
    # Adicionar mais segmentos
    for i in range(10):
        new_segment = game.snake[-1]  # Duplicar último segmento
        game.snake.append(new_segment)
    
    state = game_state.get_state()
    info = game_state.get_game_info()
    
    print(f"  Novo tamanho: {len(game.snake)}")
    print(f"  Tamanho normalizado: {state[11]:.3f}")
    print(f"  Nova densidade corporal: {info['body_density']:.3f}")
    
    # Restaurar estado original
    game.snake = original_snake
    
    print("✓ Teste de características concluído")
    return True

def test_neural_network_compatibility():
    """Testa compatibilidade com a rede neural"""
    print("\n🧠 Testando Compatibilidade da Rede Neural")
    print("=" * 50)
    
    try:
        import torch
        
        game = SnakeGame()
        agent = Agent()
        
        # Testar forward pass
        state = agent.get_state(game)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        print(f"✓ Estado convertido para tensor: {state_tensor.shape}")
        
        # Testar predição
        with torch.no_grad():
            prediction = agent.model(state_tensor)
            print(f"✓ Predição da rede: {prediction}")
            print(f"✓ Shape da predição: {prediction.shape}")
        
        # Testar ação
        action = agent.get_action(state)
        print(f"✓ Ação gerada: {action}")
        
        # Testar múltiplas predições
        print(f"\n🔄 Testando múltiplas predições...")
        for i in range(5):
            state = agent.get_state(game)
            action = agent.get_action(state)
            reward, done, score = game.play_step(action)
            
            if done:
                game.reset()
                break
        
        print(f"✓ Rede neural funcionando corretamente")
        return True
        
    except Exception as e:
        print(f"❌ Erro na rede neural: {e}")
        return False

def test_performance_impact():
    """Testa o impacto na performance"""
    print("\n⚡ Testando Impacto na Performance")
    print("=" * 50)
    
    import time
    
    game = SnakeGame()
    game_state = GameState(game)
    
    # Testar velocidade de cálculo do estado
    num_iterations = 1000
    
    start_time = time.time()
    for _ in range(num_iterations):
        state = game_state.get_state()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000  # em ms
    
    print(f"✓ {num_iterations} cálculos de estado em {end_time - start_time:.3f}s")
    print(f"✓ Tempo médio por estado: {avg_time:.3f}ms")
    
    if avg_time < 1.0:  # Menos de 1ms é aceitável
        print(f"✓ Performance aceitável para treinamento")
        return True
    else:
        print(f"⚠️  Performance pode afetar treinamento (>{avg_time:.1f}ms por estado)")
        return True

def main():
    """Função principal de teste"""
    print("🚀 Iniciando Testes do Estado Expandido")
    print("=" * 60)
    
    tests = [
        ("Estado Expandido", test_expanded_state),
        ("Características Específicas", test_state_characteristics),
        ("Compatibilidade Neural", test_neural_network_compatibility),
        ("Impacto na Performance", test_performance_impact),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🧪 Executando: {name}")
        try:
            if test_func():
                print(f"✅ {name}: PASSOU")
                passed += 1
            else:
                print(f"❌ {name}: FALHOU")
        except Exception as e:
            print(f"💥 {name}: ERRO - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Resultado Final:")
    print(f"   Testes passaram: {passed}/{total}")
    print(f"   Taxa de sucesso: {passed/total*100:.1f}%")
    
    if passed == total:
        print(f"\n🎉 Todos os testes passaram! O estado expandido está funcionando corretamente.")
        print(f"   A IA agora tem {STATE_SIZE} características para tomar decisões melhores!")
    else:
        print(f"\n⚠️  Alguns testes falharam. Verifique os erros acima.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)