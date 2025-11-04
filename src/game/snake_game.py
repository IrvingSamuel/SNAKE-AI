import pygame
import random
from enum import Enum
from collections import namedtuple
from .constants import *

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGame:
    def __init__(self, width=GAME_WIDTH, height=GAME_HEIGHT):
        self.width = width
        self.height = height
        self.grid_width = width // GRID_SIZE
        self.grid_height = height // GRID_SIZE
        
        # Inicializar pygame
        pygame.init()
        self.display = None
        self.clock = pygame.time.Clock()
        
        # Estado do jogo
        self.reset()
        
    def reset(self):
        # Estado inicial da cobrinha
        self.direction = Direction.RIGHT
        self.head = Point(self.grid_width//2, self.grid_height//2)
        self.snake = [self.head,
                     Point(self.head.x-1, self.head.y),
                     Point(self.head.x-2, self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.previous_head = None  # Para calcular eficiência do movimento
        
    def _place_food(self):
        x = random.randint(0, self.grid_width-1)
        y = random.randint(0, self.grid_height-1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
            
    def play_step(self, action=None):
        self.frame_iteration += 1
        
        # 1. Coletar entrada do usuário apenas se não há ação da IA
        if action is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                    elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                        self.direction = Direction.DOWN
        elif action != "manual":
            # Se há ação da IA, ainda precisamos processar eventos do sistema
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        # Se action == "manual", os eventos já foram processados pela interface
        
        # 2. Mover
        if action is not None and action != "manual":
            self._move_ai(action)
            # Após definir a nova direção, executar o movimento
            self._move(self.direction)
        else:
            self._move(self.direction)
        self.snake.insert(0, self.head)
        
        # 3. Verificar se o jogo acabou
        reward = 0
        game_over = False
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = REWARD_DEATH
            return reward, game_over, self.score
            
        # 4. Colocar nova comida ou apenas mover
        if self.head == self.food:
            self.score += 1
            reward = REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()
            
        # 5. Atualizar UI e clock
        if self.display is not None:
            self._update_ui()
            self.clock.tick(GAME_SPEED)
        
        # 6. Retornar game over e score
        return reward, game_over, self.score
    
    def _move_ai(self, action):
        # action pode ser uma lista [0,0,1] ou um número
        if isinstance(action, list):
            # Converter lista para índice
            action = action.index(1) if 1 in action else 0
        
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:  # straight
            new_dir = clock_wise[idx]
        elif action == 1:  # right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Bater na borda
        if pt.x >= self.grid_width or pt.x < 0 or pt.y >= self.grid_height or pt.y < 0:
            return True
        # Bater em si mesma
        if pt in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        self.display.fill(BACKGROUND_COLOR)
        
        # Desenhar grade
        for x in range(0, self.width, GRID_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, GRID_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (0, y), (self.width, y))
        
        # Desenhar cobrinha
        for pt in self.snake:
            pygame.draw.rect(self.display, SNAKE_COLOR, 
                           pygame.Rect(pt.x*GRID_SIZE, pt.y*GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
        # Desenhar comida
        pygame.draw.rect(self.display, FOOD_COLOR, 
                        pygame.Rect(self.food.x*GRID_SIZE, self.food.y*GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # Mostrar score
        font = pygame.font.Font(None, FONT_SIZE_MEDIUM)
        text = font.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.display.blit(text, [0, 0])
        
        pygame.display.flip()
        
    def _move(self, direction):
        # Armazenar posição anterior da cabeça
        self.previous_head = self.head
        
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += 1
        elif direction == Direction.LEFT:
            x -= 1
        elif direction == Direction.DOWN:
            y += 1
        elif direction == Direction.UP:
            y -= 1
            
        self.head = Point(x, y)
        
    def set_display(self, display):
        """Define o display para renderização"""
        self.display = display
