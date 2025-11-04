import numpy as np
from collections import namedtuple
from .snake_game import SnakeGame, Direction, Point
from .constants import GRID_WIDTH, GRID_HEIGHT

class GameState:
    def __init__(self, game):
        self.game = game
        
    def get_state(self):
        """
        Retorna o estado expandido atual do jogo como um array numpy
        Estado expandido com 28 características:
        - Perigos imediatos (3)
        - Direção atual (4) 
        - Localização da comida (4)
        - Tamanho da cobra normalizado (1)
        - Densidade corporal local (4)
        - Distâncias até bordas (4)
        - Espaços livres em direções (4)
        - Detecção de armadilhas (2)
        - Informações de movimento (2)
        """
        head = self.game.head
        
        # Pontos adjacentes
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        # Direções atuais
        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        # === PERIGOS IMEDIATOS (3) ===
        danger_straight = (dir_r and self.game.is_collision(point_r)) or \
                         (dir_l and self.game.is_collision(point_l)) or \
                         (dir_u and self.game.is_collision(point_u)) or \
                         (dir_d and self.game.is_collision(point_d))

        danger_right = (dir_u and self.game.is_collision(point_r)) or \
                      (dir_d and self.game.is_collision(point_l)) or \
                      (dir_l and self.game.is_collision(point_u)) or \
                      (dir_r and self.game.is_collision(point_d))

        danger_left = (dir_d and self.game.is_collision(point_r)) or \
                     (dir_u and self.game.is_collision(point_l)) or \
                     (dir_r and self.game.is_collision(point_u)) or \
                     (dir_l and self.game.is_collision(point_d))

        # === DIREÇÃO ATUAL (4) ===
        direction_left = dir_l
        direction_right = dir_r
        direction_up = dir_u
        direction_down = dir_d

        # === LOCALIZAÇÃO DA COMIDA (4) ===
        food_left = self.game.food.x < head.x
        food_right = self.game.food.x > head.x
        food_up = self.game.food.y < head.y
        food_down = self.game.food.y > head.y

        # === TAMANHO DA COBRA NORMALIZADO (1) ===
        max_possible_length = GRID_WIDTH * GRID_HEIGHT
        snake_length_normalized = len(self.game.snake) / max_possible_length

        # === DENSIDADE CORPORAL LOCAL (4) ===
        # Verifica quantos segmentos da cobra estão próximos em cada direção
        body_density_left = self._count_body_segments_in_direction(head, (-1, 0), 3)
        body_density_right = self._count_body_segments_in_direction(head, (1, 0), 3)
        body_density_up = self._count_body_segments_in_direction(head, (0, -1), 3)
        body_density_down = self._count_body_segments_in_direction(head, (0, 1), 3)

        # === DISTÂNCIAS ATÉ BORDAS (4) ===
        distance_to_left_wall = head.x / GRID_WIDTH
        distance_to_right_wall = (GRID_WIDTH - 1 - head.x) / GRID_WIDTH
        distance_to_top_wall = head.y / GRID_HEIGHT
        distance_to_bottom_wall = (GRID_HEIGHT - 1 - head.y) / GRID_HEIGHT

        # === ESPAÇOS LIVRES EM DIREÇÕES (4) ===
        free_spaces_left = self._count_free_spaces_in_direction(head, (-1, 0))
        free_spaces_right = self._count_free_spaces_in_direction(head, (1, 0))
        free_spaces_up = self._count_free_spaces_in_direction(head, (0, -1))
        free_spaces_down = self._count_free_spaces_in_direction(head, (0, 1))

        # === DETECÇÃO DE ARMADILHAS (2) ===
        # Verifica se a cobra pode ficar presa
        tail_blocking_escape = self._is_tail_blocking_escape()
        potential_trap = self._detect_potential_trap()

        # === INFORMAÇÕES DE MOVIMENTO (2) ===
        # Distância atual até a comida
        distance_to_food = (abs(head.x - self.game.food.x) + abs(head.y - self.game.food.y)) / (GRID_WIDTH + GRID_HEIGHT)
        
        # Eficiência de movimento (se está se aproximando da comida comparado ao movimento anterior)
        movement_efficiency = self._calculate_movement_efficiency()

        # Compilar estado final
        state = [
            # Perigos imediatos (3)
            danger_straight,
            danger_right,
            danger_left,
            
            # Direção atual (4)
            direction_left,
            direction_right,
            direction_up,
            direction_down,
            
            # Localização da comida (4)
            food_left,
            food_right,
            food_up,
            food_down,
            
            # Tamanho da cobra (1)
            snake_length_normalized,
            
            # Densidade corporal local (4)
            body_density_left,
            body_density_right,
            body_density_up,
            body_density_down,
            
            # Distâncias até bordas (4)
            distance_to_left_wall,
            distance_to_right_wall,
            distance_to_top_wall,
            distance_to_bottom_wall,
            
            # Espaços livres (4)
            free_spaces_left,
            free_spaces_right,
            free_spaces_up,
            free_spaces_down,
            
            # Detecção de armadilhas (2)
            tail_blocking_escape,
            potential_trap,
            
            # Informações de movimento (2)
            distance_to_food,
            movement_efficiency
        ]

        return np.array(state, dtype=np.float32)
    
    def _count_body_segments_in_direction(self, start_point, direction, max_distance):
        """Conta quantos segmentos do corpo estão em uma direção específica"""
        count = 0
        dx, dy = direction
        
        for i in range(1, max_distance + 1):
            check_point = Point(start_point.x + dx * i, start_point.y + dy * i)
            
            # Verificar se está dentro dos limites
            if (0 <= check_point.x < GRID_WIDTH and 0 <= check_point.y < GRID_HEIGHT):
                # Verificar se há um segmento do corpo nesta posição
                if check_point in self.game.snake[1:]:  # Excluir a cabeça
                    count += 1
            else:
                break
        
        return count / max_distance  # Normalizar
    
    def _count_free_spaces_in_direction(self, start_point, direction):
        """Conta quantos espaços livres existem em uma direção até encontrar um obstáculo"""
        count = 0
        dx, dy = direction
        
        current_point = Point(start_point.x + dx, start_point.y + dy)
        
        while (0 <= current_point.x < GRID_WIDTH and 
               0 <= current_point.y < GRID_HEIGHT and
               not self.game.is_collision(current_point)):
            count += 1
            current_point = Point(current_point.x + dx, current_point.y + dy)
        
        # Normalizar pelo tamanho máximo possível na direção
        max_possible = max(GRID_WIDTH, GRID_HEIGHT)
        return count / max_possible
    
    def _is_tail_blocking_escape(self):
        """Verifica se a cauda está bloqueando uma rota de escape"""
        head = self.game.head
        tail = self.game.snake[-1]
        
        # Se a cobra for muito pequena, a cauda não é um problema
        if len(self.game.snake) < 4:
            return False
        
        # Verificar se a cauda está próxima da cabeça (dentro de 3 posições)
        distance_to_tail = abs(head.x - tail.x) + abs(head.y - tail.y)
        
        if distance_to_tail <= 3:
            # Verificar se a cauda está bloqueando um caminho direto para a comida
            food = self.game.food
            
            # Se a cauda está entre a cabeça e a comida
            if ((head.x <= tail.x <= food.x or food.x <= tail.x <= head.x) and
                (head.y <= tail.y <= food.y or food.y <= tail.y <= head.y)):
                return True
        
        return False
    
    def _detect_potential_trap(self):
        """Detecta se a cobra pode ficar presa em um espaço pequeno"""
        head = self.game.head
        
        # Contar espaços livres ao redor da cabeça
        free_adjacent = 0
        adjacent_points = [
            Point(head.x - 1, head.y),
            Point(head.x + 1, head.y),
            Point(head.x, head.y - 1),
            Point(head.x, head.y + 1)
        ]
        
        for point in adjacent_points:
            if (0 <= point.x < GRID_WIDTH and 
                0 <= point.y < GRID_HEIGHT and
                not self.game.is_collision(point)):
                free_adjacent += 1
        
        # Se há apenas 1 ou 0 espaços livres, é uma armadilha
        if free_adjacent <= 1:
            return True
        
        # Verificar se está em um corredor estreito
        if free_adjacent == 2:
            # Calcular o "espaço de manobra" - quantos movimentos livres existem
            total_free_space = 0
            for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                total_free_space += self._count_free_spaces_in_direction(head, direction) * 10
            
            # Se o espaço total é muito limitado comparado ao tamanho da cobra
            if total_free_space < len(self.game.snake) * 0.5:
                return True
        
        return False
    
    def _calculate_movement_efficiency(self):
        """Calcula a eficiência do movimento em direção à comida"""
        head = self.game.head
        food = self.game.food
        
        # Distância atual
        current_distance = abs(head.x - food.x) + abs(head.y - food.y)
        
        # Se não há histórico, retorna neutro
        if not hasattr(self.game, 'previous_head') or self.game.previous_head is None:
            return 0.5
        
        # Distância anterior
        prev_head = self.game.previous_head
        previous_distance = abs(prev_head.x - food.x) + abs(prev_head.y - food.y)
        
        # Calcular eficiência
        if previous_distance == 0:
            return 1.0  # Estava na comida
        
        if current_distance < previous_distance:
            return 1.0  # Aproximou-se
        elif current_distance > previous_distance:
            return 0.0  # Afastou-se
        else:
            return 0.5  # Manteve distância
    
    def get_distance_to_food(self):
        """Calcula a distância Manhattan até a comida"""
        return abs(self.game.head.x - self.game.food.x) + abs(self.game.head.y - self.game.food.y)
    
    def is_moving_closer_to_food(self, old_distance, new_distance):
        """Verifica se a cobrinha está se movendo em direção à comida"""
        return new_distance < old_distance
    
    def get_game_info(self):
        """Retorna informações úteis sobre o estado do jogo"""
        return {
            'score': self.game.score,
            'snake_length': len(self.game.snake),
            'food_position': self.game.food,
            'head_position': self.game.head,
            'direction': self.game.direction,
            'frame_iteration': self.game.frame_iteration,
            'distance_to_food': self.get_distance_to_food(),
            'state_size': 28,  # Novo tamanho do estado
            'body_density': sum([
                self._count_body_segments_in_direction(self.game.head, (-1, 0), 3),
                self._count_body_segments_in_direction(self.game.head, (1, 0), 3),
                self._count_body_segments_in_direction(self.game.head, (0, -1), 3),
                self._count_body_segments_in_direction(self.game.head, (0, 1), 3)
            ]) / 4,
            'free_space_ratio': sum([
                self._count_free_spaces_in_direction(self.game.head, (-1, 0)),
                self._count_free_spaces_in_direction(self.game.head, (1, 0)),
                self._count_free_spaces_in_direction(self.game.head, (0, -1)),
                self._count_free_spaces_in_direction(self.game.head, (0, 1))
            ]) / 4,
            'trap_risk': self._detect_potential_trap(),
            'tail_blocking': self._is_tail_blocking_escape()
        }
