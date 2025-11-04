import pygame
import sys
import threading
import time
from ..ai.training import Trainer
from ..interface.visualization import RealTimeMonitor
from ..game.constants import *

class TrainingInterface:
    """Interface dedicada para o treinamento da IA com visualizações"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake AI - Treinamento")
        self.clock = pygame.time.Clock()
        
        # Inicializar trainer
        self.trainer = Trainer()
        
        # Tentar carregar o último modelo automaticamente
        self.trainer.agent.auto_load_latest()
        
        self.monitor = None
        
        # Estado da interface
        self.training = False
        self.running = True
        self.paused = False
        
        # Fontes
        self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
        self.font_medium = pygame.font.Font(None, FONT_SIZE_MEDIUM)
        self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)
        
        # Thread de treinamento
        self.training_thread = None
        
    def handle_events(self):
        """Processa eventos do pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop_training()
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.stop_training()
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_t and not self.training:
                    self.start_training()
                elif event.key == pygame.K_s and self.training:
                    self.stop_training()
                elif event.key == pygame.K_r:
                    self.reset_training()
                elif event.key == pygame.K_m:
                    self.start_monitoring()
    
    def start_training(self):
        """Inicia o treinamento em thread separada"""
        if not self.training:
            self.training = True
            self.training_thread = threading.Thread(target=self._training_loop)
            self.training_thread.daemon = True
            self.training_thread.start()
            print("Treinamento iniciado!")
    
    def stop_training(self):
        """Para o treinamento"""
        if self.training:
            self.training = False
            if self.monitor:
                self.monitor.stop_monitoring()
            print("Treinamento parado!")
    
    def reset_training(self):
        """Reseta o treinamento"""
        self.stop_training()
        self.trainer = Trainer()
        print("Treinamento resetado!")
    
    def start_monitoring(self):
        """Inicia o monitor de visualização"""
        if not self.monitor and self.training:
            self.monitor = RealTimeMonitor(self.trainer.agent)
            self.monitor.start_monitoring()
            print("Monitor de visualização iniciado!")
    
    def _training_loop(self):
        """Loop de treinamento executado em thread separada"""
        game_count = 0
        max_games = 10000  # Pode ser configurado
        
        while self.training and game_count < max_games:
            if self.paused:
                time.sleep(0.1)
                continue
                
            # Estado atual
            state_old = self.trainer.agent.get_state(self.trainer.game)
            
            # Obter ação
            final_move = self.trainer.agent.get_action(state_old)
            
            # Executar movimento
            reward, done, score = self.trainer.game.play_step(final_move)
            state_new = self.trainer.agent.get_state(self.trainer.game)
            
            # Treinar
            self.trainer.agent.train_short_memory(state_old, final_move, reward, state_new, done)
            self.trainer.agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                # Fim do jogo
                self.trainer.game.reset()
                self.trainer.agent.n_games += 1
                self.trainer.agent.train_long_memory()
                
                if score > self.trainer.agent.record:
                    self.trainer.agent.record = score
                
                self.trainer.agent.scores.append(score)
                self.trainer.agent.total_score += score
                mean_score = self.trainer.agent.total_score / self.trainer.agent.n_games
                self.trainer.agent.mean_scores.append(mean_score)
                
                game_count = self.trainer.agent.n_games
                
                # Atualizar monitor
                if self.monitor:
                    self.monitor.update()
                
                # Salvar modelo periodicamente
                if game_count % 100 == 0:
                    self.trainer.agent.save_model(f'models/checkpoint_{game_count}.pth')
                
                # Pequena pausa para não sobrecarregar
                time.sleep(0.001)
        
        # Salvar modelo final
        if self.training:
            self.trainer.agent.save_model('models/final_model.pth')
            if self.monitor:
                self.monitor.save_report()
        
        self.training = False
    
    def draw_training_info(self):
        """Desenha informações do treinamento"""
        self.screen.fill(BLACK)
        
        y_offset = 50
        
        # Título
        title = self.font_large.render("Snake AI - Interface de Treinamento", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH//2, y_offset))
        self.screen.blit(title, title_rect)
        y_offset += 80
        
        # Status
        if self.training:
            status = "TREINANDO"
            color = GREEN
        else:
            status = "PARADO"
            color = RED
            
        if self.paused:
            status += " (PAUSADO)"
            color = YELLOW
            
        status_text = self.font_medium.render(f"Status: {status}", True, color)
        self.screen.blit(status_text, (50, y_offset))
        y_offset += 40
        
        # Estatísticas
        stats = self.trainer.agent.get_stats()
        
        stats_info = [
            f"Jogos Completados: {stats['games_played']}",
            f"Record: {stats['record_score']}",
            f"Média dos Últimos 100: {stats['mean_score']:.1f}",
            f"Taxa de Exploração: {stats['epsilon']:.3f}",
            f"Score Total: {stats['total_score']}"
        ]
        
        for info in stats_info:
            info_text = self.font_medium.render(info, True, WHITE)
            self.screen.blit(info_text, (50, y_offset))
            y_offset += 30
        
        y_offset += 40
        
        # Últimos scores
        if stats['recent_scores']:
            recent_title = self.font_medium.render("Últimos 10 Scores:", True, YELLOW)
            self.screen.blit(recent_title, (50, y_offset))
            y_offset += 30
            
            scores_text = ", ".join(map(str, stats['recent_scores']))
            scores_render = self.font_small.render(scores_text, True, WHITE)
            self.screen.blit(scores_render, (50, y_offset))
            y_offset += 50
        
        # Controles
        controls_title = self.font_medium.render("Controles:", True, YELLOW)
        self.screen.blit(controls_title, (50, y_offset))
        y_offset += 30
        
        controls = [
            "T - Iniciar Treinamento",
            "S - Parar Treinamento", 
            "SPACE - Pausar/Continuar",
            "R - Resetar Treinamento",
            "M - Iniciar Monitor Visual",
            "ESC - Sair"
        ]
        
        for control in controls:
            control_text = self.font_small.render(control, True, WHITE)
            self.screen.blit(control_text, (50, y_offset))
            y_offset += 25
        
        # Progresso visual simples
        if self.training and stats['games_played'] > 0:
            progress_y = WINDOW_HEIGHT - 100
            progress_width = WINDOW_WIDTH - 100
            progress_height = 20
            
            # Barra de progresso baseada no epsilon (quanto menor, mais treinado)
            progress_percent = 1 - (stats['epsilon'] / 1.0)
            
            # Fundo da barra
            pygame.draw.rect(self.screen, DARK_GRAY, 
                           (50, progress_y, progress_width, progress_height))
            
            # Progresso
            pygame.draw.rect(self.screen, GREEN, 
                           (50, progress_y, progress_width * progress_percent, progress_height))
            
            # Texto do progresso
            progress_text = self.font_small.render(f"Progresso: {progress_percent*100:.1f}%", True, WHITE)
            self.screen.blit(progress_text, (50, progress_y + 25))
    
    def run(self):
        """Loop principal da interface de treinamento"""
        print("Interface de Treinamento iniciada!")
        print("Controles: T=Treinar, S=Parar, SPACE=Pausar, R=Reset, M=Monitor, ESC=Sair")
        
        while self.running:
            self.handle_events()
            self.draw_training_info()
            pygame.display.flip()
            self.clock.tick(30)  # FPS moderado para a interface
        
        self.stop_training()
        pygame.quit()
        sys.exit()
