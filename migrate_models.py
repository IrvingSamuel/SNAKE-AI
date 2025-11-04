#!/usr/bin/env python3
"""
Migração de Modelos Antigos
===========================

Este script converte modelos antigos (LinearQNet com 11 características)
para o novo formato (DQN com 28 características).

Como os modelos antigos têm menos entradas, não podemos fazer uma conversão
direta. Em vez disso, salvamos as estatísticas de treinamento e reiniciamos
o treinamento com a nova arquitetura.

"""

import os
import sys
import glob
import torch
from datetime import datetime

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ai.neural_network import LinearQNet, DQN
from src.game.constants import *

def convert_old_model(old_file_path, output_dir='models/migrated'):
    """Converte um modelo antigo para estatísticas de treinamento"""
    
    print(f"Processando: {old_file_path}")
    
    try:
        # Carregar checkpoint antigo
        checkpoint = torch.load(old_file_path, weights_only=False)
        
        # Extrair informações
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint completo
            stats = {
                'original_file': old_file_path,
                'n_games': checkpoint.get('n_games', 0),
                'record': checkpoint.get('record', 0),
                'scores': checkpoint.get('scores', []),
                'mean_scores': checkpoint.get('mean_scores', []),
                'total_score': checkpoint.get('total_score', 0),
                'epsilon': checkpoint.get('epsilon', EPSILON_END),
                'migration_date': datetime.now().isoformat(),
                'old_state_size': 11,
                'new_state_size': STATE_SIZE,
                'old_hidden_size': 256,
                'new_hidden_size': HIDDEN_SIZE
            }
            
            # Criar diretório de saída
            os.makedirs(output_dir, exist_ok=True)
            
            # Nome do arquivo de estatísticas
            base_name = os.path.basename(old_file_path).replace('.pth', '')
            stats_file = os.path.join(output_dir, f"{base_name}_stats.json")
            
            # Salvar estatísticas
            import json
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"  ✓ Estatísticas salvas: {stats_file}")
            print(f"    Jogos: {stats['n_games']}, Record: {stats['record']}")
            
            return stats
        
        else:
            print(f"  ⚠️ Não é um checkpoint completo: {old_file_path}")
            return None
            
    except Exception as e:
        print(f"  ❌ Erro ao processar {old_file_path}: {e}")
        return None

def find_best_old_model():
    """Encontra o melhor modelo antigo para usar como base"""
    
    model_patterns = [
        'models/checkpoint_*.pth',
        'models/model_*.pth'
    ]
    
    best_stats = None
    best_file = None
    
    for pattern in model_patterns:
        files = glob.glob(pattern)
        for file in files:
            stats = convert_old_model(file)
            if stats and (best_stats is None or stats['record'] > best_stats['record']):
                best_stats = stats
                best_file = file
    
    return best_file, best_stats

def create_migration_checkpoint(stats, output_file='models/migration_checkpoint.pth'):
    """Cria um checkpoint de migração com a nova arquitetura"""
    
    # Criar novo modelo DQN
    new_model = DQN(STATE_SIZE, HIDDEN_SIZE, ACTION_SIZE)
    
    # Criar checkpoint com estatísticas migradas
    checkpoint = {
        'model_state_dict': new_model.state_dict(),
        'n_games': 0,  # Resetar jogos mas manter record como meta
        'epsilon': EPSILON_START,  # Resetar epsilon
        'record': stats.get('record', 0),  # Manter record como meta
        'scores': [],  # Resetar scores
        'mean_scores': [],  # Resetar mean scores
        'total_score': 0,  # Resetar total
        'gamma': GAMMA,
        'learning_rate': LEARNING_RATE,
        'migration_info': {
            'migrated_from': stats.get('original_file', 'unknown'),
            'original_record': stats.get('record', 0),
            'original_games': stats.get('n_games', 0),
            'migration_date': datetime.now().isoformat(),
            'old_architecture': f"LinearQNet({stats.get('old_state_size', 11)}, {stats.get('old_hidden_size', 256)}, {ACTION_SIZE})",
            'new_architecture': f"DQN({STATE_SIZE}, {HIDDEN_SIZE}, {ACTION_SIZE})"
        }
    }
    
    # Salvar checkpoint
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(checkpoint, output_file)
    
    print(f"✓ Checkpoint de migração criado: {output_file}")
    print(f"  Record anterior: {stats.get('record', 0)}")
    print(f"  Jogos anteriores: {stats.get('n_games', 0)}")
    print(f"  Nova arquitetura: DQN com {STATE_SIZE} características")
    
    return output_file

def main():
    """Função principal de migração"""
    
    print("🔄 Migração de Modelos Snake AI")
    print("=" * 50)
    print(f"Convertendo de LinearQNet (11 características) para DQN ({STATE_SIZE} características)")
    print()
    
    # Encontrar melhor modelo antigo
    print("📊 Procurando modelos antigos...")
    best_file, best_stats = find_best_old_model()
    
    if not best_stats:
        print("❌ Nenhum modelo antigo válido encontrado.")
        print("💡 Iniciando treinamento do zero com nova arquitetura.")
        return
    
    print(f"\n🏆 Melhor modelo encontrado:")
    print(f"   Arquivo: {best_file}")
    print(f"   Record: {best_stats['record']}")
    print(f"   Jogos: {best_stats['n_games']}")
    
    # Criar checkpoint de migração
    print(f"\n🔧 Criando checkpoint de migração...")
    migration_file = create_migration_checkpoint(best_stats)
    
    print(f"\n✅ Migração concluída!")
    print(f"Use o arquivo: {migration_file}")
    print()
    print(f"🚀 Para usar o modelo migrado:")
    print(f"   python main.py --mode train --model {migration_file}")
    print()
    print(f"🎯 Meta de performance:")
    print(f"   Superar record anterior de {best_stats['record']} pontos")
    print(f"   Com {STATE_SIZE} características, esperamos melhor performance!")

if __name__ == "__main__":
    main()