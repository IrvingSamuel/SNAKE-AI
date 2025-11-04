import pygame
import sys
from ..game.snake_game import SnakeGame, Direction
from ..game.constants import *
from ..ai.agent import Agent

class GameInterface:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake AI - Jogo da Cobrinha com IA")
        self.clock = pygame.time.Clock()
        
        # Configurar áreas da tela
        self.game_surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.info_surface = pygame.Surface((INFO_PANEL_WIDTH, WINDOW_HEIGHT))
        
        # Inicializar jogo e IA
        self.game = SnakeGame(GAME_WIDTH, GAME_HEIGHT)
        self.game.set_display(self.game_surface)
        self.agent = Agent()
        
        # Tentar carregar o último modelo automaticamente
        self.agent.auto_load_latest()
        
        # Estado do jogo
        self.mode = "MANUAL"  # MANUAL, AI_PLAY, AI_TRAIN
        self.paused = False
        self.running = True
        
        # Fontes
        self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
        self.font_medium = pygame.font.Font(None, FONT_SIZE_MEDIUM)
        self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)
        
    def handle_events(self):
        """Processa eventos do pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.game.reset()
                elif event.key == pygame.K_m:
                    self.mode = "MANUAL"
                    self.game.reset()
                elif event.key == pygame.K_a:
                    self.mode = "AI_PLAY"
                    self.game.reset()
                elif event.key == pygame.K_t:
                    self.mode = "AI_TRAIN"
                    self.game.reset()
                elif event.key == pygame.K_s and pygame.key.get_pressed()[pygame.K_LCTRL]:
                    # Salvar modelo atual
                    success = self.agent.save_model('models/manual_save.pth')
                    if success:
                        print("Modelo salvo manualmente!")
                elif event.key == pygame.K_l and pygame.key.get_pressed()[pygame.K_LCTRL]:
                    # Carregar último modelo automaticamente
                    success = self.agent.auto_load_latest()
                    if success:
                        print("Último modelo carregado!")
                elif event.key == pygame.K_n and pygame.key.get_pressed()[pygame.K_LCTRL]:
                    # Novo treinamento (reset do agente)
                    self.agent = Agent()
                    print("Novo agente criado - treinamento do zero!")
                # Processar controles de movimento no modo manual
                elif self.mode == "MANUAL":
                    if event.key == pygame.K_LEFT and self.game.direction != Direction.RIGHT:
                        self.game.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT and self.game.direction != Direction.LEFT:
                        self.game.direction = Direction.RIGHT
                    elif event.key == pygame.K_UP and self.game.direction != Direction.DOWN:
                        self.game.direction = Direction.UP
                    elif event.key == pygame.K_DOWN and self.game.direction != Direction.UP:
                        self.game.direction = Direction.DOWN

    def draw_info_panel(self):
        """Desenha o painel de informações"""
        self.info_surface.fill(PANEL_COLOR)
        y_offset = 20
        
        # Título
        title = self.font_large.render("Snake AI", True, WHITE)
        self.info_surface.blit(title, (20, y_offset))
        y_offset += 50
        
        # Modo atual
        mode_text = self.font_medium.render(f"Modo: {self.mode}", True, WHITE)
        self.info_surface.blit(mode_text, (20, y_offset))
        y_offset += 30
        
        # Score
        score_text = self.font_medium.render(f"Score: {self.game.score}", True, WHITE)
        self.info_surface.blit(score_text, (20, y_offset))
        y_offset += 30
        
        # Comprimento da cobrinha
        length_text = self.font_medium.render(f"Tamanho: {len(self.game.snake)}", True, WHITE)
        self.info_surface.blit(length_text, (20, y_offset))
        y_offset += 50
        
        # Estatísticas da IA
        if self.mode in ["AI_PLAY", "AI_TRAIN"]:
            stats = self.agent.get_stats()
            
            ai_title = self.font_medium.render("Estatísticas IA:", True, YELLOW)
            self.info_surface.blit(ai_title, (20, y_offset))
            y_offset += 30
            
            games_text = self.font_small.render(f"Jogos: {stats['games_played']}", True, WHITE)
            self.info_surface.blit(games_text, (20, y_offset))
            y_offset += 25
            
            record_text = self.font_small.render(f"Record: {stats['record_score']}", True, WHITE)
            self.info_surface.blit(record_text, (20, y_offset))
            y_offset += 25
            
            mean_text = self.font_small.render(f"Média: {stats['mean_score']:.1f}", True, WHITE)
            self.info_surface.blit(mean_text, (20, y_offset))
            y_offset += 25
            
            epsilon_text = self.font_small.render(f"Epsilon: {stats['epsilon']:.3f}", True, WHITE)
            self.info_surface.blit(epsilon_text, (20, y_offset))
            y_offset += 40
        
        # Controles
        controls_title = self.font_medium.render("Controles:", True, YELLOW)
        self.info_surface.blit(controls_title, (20, y_offset))
        y_offset += 30
        
        controls = [
            "SPACE - Pausar",
            "R - Resetar jogo",
            "M - Modo Manual",
            "A - IA Jogar",
            "T - IA Treinar",
            "CTRL+S - Salvar modelo",
            "CTRL+L - Carregar último modelo",
            "CTRL+N - Novo treinamento",
            "ESC - Sair"
        ]
        
        for control in controls:
            control_text = self.font_small.render(control, True, WHITE)
            self.info_surface.blit(control_text, (20, y_offset))
            y_offset += 20
            
        # Status
        y_offset += 20
        if self.paused:
            status_text = self.font_medium.render("PAUSADO", True, RED)
            self.info_surface.blit(status_text, (20, y_offset))
        
    def update_game(self):
        """Atualiza o estado do jogo baseado no modo atual"""
        if self.paused:
            return
            
        if self.mode == "MANUAL":
            # Modo manual - movimento controlado pelos eventos, só executar um step de jogo
            reward, game_over, score = self.game.play_step(action="manual")
            
        elif self.mode == "AI_PLAY":
            # IA jogando com modelo treinado
            state = self.agent.get_state(self.game)
            action = self.agent.get_action(state)
            reward, game_over, score = self.game.play_step(action)
            
        elif self.mode == "AI_TRAIN":
            # IA treinando
            state_old = self.agent.get_state(self.game)
            action = self.agent.get_action(state_old)
            reward, game_over, score = self.game.play_step(action)
            state_new = self.agent.get_state(self.game)
            
            # Treinar
            self.agent.train_short_memory(state_old, action, reward, state_new, game_over)
            self.agent.remember(state_old, action, reward, state_new, game_over)
            
            if game_over:
                self.agent.n_games += 1
                self.agent.train_long_memory()
                
                if score > self.agent.record:
                    self.agent.record = score
                    
                self.agent.scores.append(score)
                self.agent.total_score += score
                mean_score = self.agent.total_score / self.agent.n_games
                self.agent.mean_scores.append(mean_score)
                
                # Auto-salvar a cada 100 jogos
                if self.agent.n_games % 100 == 0:
                    self.agent.save_model(f'models/checkpoint_{self.agent.n_games}.pth')
                
                # Salvar backup a cada 500 jogos
                if self.agent.n_games % 500 == 0:
                    self.agent.save_model(f'models/backup_{self.agent.n_games}.pth')
                
                # Salvar modelo final quando atinge novo record
                if score == self.agent.record and score > 5:
                    self.agent.save_model('models/model_best.pth')
        
        # Reset automático quando o jogo acaba (com pequena pausa)
        if game_over and self.mode != "MANUAL":
            self.game.reset()
            
            # Log do progresso a cada 50 jogos no treinamento
            if self.mode == "AI_TRAIN" and self.agent.n_games % 50 == 0:
                print(f'Jogo {self.agent.n_games}, Score: {score}, Record: {self.agent.record}, Epsilon: {self.agent.epsilon:.3f}')
    
    def draw(self):
        """Desenha toda a interface"""
        self.screen.fill(BLACK)
        
        # Desenhar jogo
        self.game._update_ui()
        self.screen.blit(self.game_surface, (0, 0))
        
        # Desenhar painel de informações
        self.draw_info_panel()
        self.screen.blit(self.info_surface, (GAME_WIDTH, 0))
        
        # Linha divisória
        pygame.draw.line(self.screen, WHITE, (GAME_WIDTH, 0), (GAME_WIDTH, WINDOW_HEIGHT), 2)
        
        pygame.display.flip()
    
    def run(self):
        """Loop principal da interface"""
        print("Interface iniciada!")
        print("Controles: M=Manual, A=IA Jogar, T=IA Treinar, R=Reset, SPACE=Pausar")
        
        while self.running:
            self.handle_events()
            self.update_game()
            self.draw()
            
            # Velocidade diferente para cada modo
            if self.mode == "AI_TRAIN":
                self.clock.tick(AI_SPEED)  # Mais rápido para treinamento
            elif self.mode == "AI_PLAY":
                self.clock.tick(GAME_SPEED * 2)  # Um pouco mais rápido para assistir
            else:
                self.clock.tick(GAME_SPEED)  # Normal para modo manual
        
        pygame.quit()
        sys.exit()
