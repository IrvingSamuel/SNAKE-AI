import os
import sys
import argparse
from datetime import datetime

def print_banner():
    """Imprime banner do projeto"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                          SNAKE AI                             ║
    ║                  Jogo da Cobrinha com IA                      ║
    ║                                                               ║
    ║              Aprendizado por Reforço (Deep Q-Learning)        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def setup_directories():
    """Cria diretórios necessários se não existirem"""
    directories = ['models', 'data', 'data/training_data', 'models/saved_models']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Diretório criado: {directory}")

def get_model_files():
    """Retorna lista de modelos salvos"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        return []
    
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.pth'):
            model_files.append(file)
    
    return sorted(model_files)

def format_time(seconds):
    """Formata tempo em segundos para formato legível"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def calculate_eta(games_completed, target_games, start_time):
    """Calcula tempo estimado para conclusão"""
    if games_completed == 0:
        return "Calculando..."
    
    elapsed_time = datetime.now().timestamp() - start_time
    time_per_game = elapsed_time / games_completed
    remaining_games = target_games - games_completed
    eta_seconds = remaining_games * time_per_game
    
    return format_time(eta_seconds)

def validate_dependencies():
    """Verifica se todas as dependências estão instaladas"""
    dependencies = {
        'pygame': 'pygame',
        'torch': 'torch', 
        'numpy': 'numpy',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - FALTANDO")
            missing.append(name)
    
    if missing:
        print(f"\nDependências faltando: {', '.join(missing)}")
        print("Execute: pip install -r requirements.txt")
        return False
    
    print("Todas as dependências estão instaladas!")
    return True

def get_system_info():
    """Retorna informações do sistema"""
    import platform
    
    info = {
        'Sistema': platform.system(),
        'Versão': platform.version(),
        'Arquitetura': platform.architecture()[0],
        'Processador': platform.processor(),
        'Python': platform.python_version()
    }
    
    # Verificar se CUDA está disponível
    try:
        import torch
        info['CUDA Disponível'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['Dispositivo CUDA'] = torch.cuda.get_device_name(0)
    except ImportError:
        info['CUDA Disponível'] = 'PyTorch não instalado'
    
    return info

def create_config_template():
    """Cria template de configuração"""
    config = {
        "game": {
            "window_width": 1400,
            "window_height": 800,
            "game_width": 600,
            "game_height": 600,
            "grid_size": 20,
            "game_speed": 10,
            "ai_speed": 30
        },
        "neural_network": {
            "state_size": 11,
            "action_size": 3,
            "hidden_size": 256,
            "learning_rate": 0.001
        },
        "training": {
            "gamma": 0.9,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "memory_size": 10000,
            "batch_size": 32,
            "target_update": 100
        },
        "rewards": {
            "food": 10,
            "death": -10,
            "move": 0,
            "closer_to_food": 1,
            "farther_from_food": -1
        }
    }
    
    return config

def log_session(mode, duration, stats=None):
    """Registra sessão em log"""
    log_dir = 'data'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'session_log.txt')
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Sessão: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modo: {mode}\n")
        f.write(f"Duração: {format_time(duration)}\n")
        
        if stats:
            f.write("Estatísticas:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
        
        f.write(f"{'='*50}\n")

class ProgressBar:
    """Barra de progresso simples para terminal"""
    
    def __init__(self, total, width=50):
        self.total = total
        self.width = width
        self.current = 0
    
    def update(self, current):
        """Atualiza progresso"""
        self.current = current
        percent = (current / self.total) * 100
        filled = int((current / self.total) * self.width)
        
        bar = '█' * filled + '░' * (self.width - filled)
        
        print(f'\r[{bar}] {percent:.1f}% ({current}/{self.total})', end='', flush=True)
    
    def finish(self):
        """Finaliza barra de progresso"""
        print()  # Nova linha
