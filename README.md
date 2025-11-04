# Snake AI - Jogo da Cobrinha com Rede Neural IA Avançada by Irving Samuel

Um jogo da cobrinha implementado em Python onde uma rede neural **avançada** aprende a jogar usando aprendizado por reforço (Deep Q-Learning). O projeto foi **significativamente melhorado** com um sistema de estado expandido e arquitetura neural mais robusta.

## 🚀 Novas Características (v2.0)

### 🧠 Sistema de Estado Expandido
- **28 características** (expandido de 11) para decisões mais inteligentes
- **Detecção de armadilhas** e planejamento espacial avançado
- **Análise de densidade corporal** e espaços livres
- **Eficiência de movimento** e otimização de trajetória

### 🏗️ Arquitetura Neural Melhorada
- **DQN de 3 camadas** com 512 neurônios (era LinearQNet com 256)
- **Sistema de recompensas inteligente** com múltiplos critérios
- **Compatibilidade com modelos antigos** e migração automática

## Características Principais

- 🐍 Jogo da cobrinha clássico implementado com PyGame
- 🧠 **Rede neural avançada DQN** que aprende usando Deep Q-Learning
- 🎯 **IA superinteligente** com 28 características de entrada
- 📊 Interface de visualização em tempo real do treinamento
- 📈 Gráficos de performance e métricas de aprendizado avançadas
- 🎮 Modo manual para jogar você mesmo
- 🤖 Modo automático para assistir a IA jogar
- 🚨 **Detecção automática de armadilhas** e situações perigosas
- 🗺️ **Planejamento espacial inteligente** e uso eficiente do espaço

## Tecnologias Utilizadas

- **Python 3.8+**
- **PyGame** - Interface gráfica do jogo
- **PyTorch** - Implementação da rede neural
- **Matplotlib** - Gráficos e visualizações
- **NumPy** - Operações matemáticas
- **OpenCV** - Processamento de imagem (opcional)

## Estrutura do Projeto

```
Snake/
├── src/
│   ├── game/
│   │   ├── __init__.py
│   │   ├── snake_game.py
│   │   ├── game_state.py
│   │   └── constants.py
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── neural_network.py
│   │   ├── agent.py
│   │   └── training.py
│   ├── interface/
│   │   ├── __init__.py
│   │   ├── game_interface.py
│   │   ├── training_interface.py
│   │   └── visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── data_processing.py
│       └── helpers.py
├── models/
│   └── saved_models/
├── data/
│   └── training_data/
├── requirements.txt
├── main.py
└── README.md
```

## Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd Snake
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o jogo:
```bash
python main.py
```

## 🎯 Performance e Resultados

### 📊 Comparação de Versões
| Versão | Características | Arquitetura | Record Máximo | Performance |
|--------|-----------------|-------------|---------------|-------------|
| v1.0 | 11 básicas | LinearQNet (256) | 85 pontos | Boa |
| **v2.0** | **28 expandidas** | **DQN (512)** | **100+** ⭐ | **Excelente** |

### 🧠 Características do Estado Expandido (28)
1. **Detecção de Perigos** (3) - Obstáculos imediatos
2. **Direção Atual** (4) - Orientação da cobra  
3. **Localização da Comida** (4) - Posição relativa do alimento
4. **Tamanho da Cobra** (1) - Comprimento normalizado
5. **Densidade Corporal** (4) - Concentração de segmentos por direção
6. **Distâncias até Bordas** (4) - Proximidade com paredes
7. **Espaços Livres** (4) - Mobilidade disponível por direção
8. **Detecção de Armadilhas** (2) - Situações perigosas
9. **Eficiência de Movimento** (2) - Otimização de trajetória

## 🚀 Como Usar

### 📥 Instalação Rápida
```bash
# Clonar repositório
git clone [url-do-repositorio]
cd Snake

# Instalar dependências
pip install -r requirements.txt

# Verificar instalação
python main.py --check-deps
```

### 🎮 Modos de Execução

#### 🤖 Treinamento da IA (Recomendado)
```bash
# Treinamento com interface visual
python main.py --mode train-interface

# Treinamento rápido por linha de comando
python main.py --mode train --games 2000

# Continuar treinamento de modelo existente
python main.py --mode train --model models/checkpoint_X.pth --games 1000
```

#### 🕹️ Jogar
```bash
# Interface completa (Manual + IA)
python main.py --mode play

# Menu interativo (mais fácil)
python main.py
```

### 🎯 Controles da Interface
- **M** - Modo manual
- **A** - IA jogar automaticamente  
- **T** - Treinar IA
- **R** - Reset do jogo
- **SPACE** - Pausar/Despausar
- **ESC** - Sair

### 🔧 Migração de Modelos Antigos
Se você tem modelos da versão anterior (v1.0):
```bash
# Migrar modelos automaticamente
python migrate_models.py

# Treinar com modelo migrado
python main.py --mode train --model models/migration_checkpoint.pth
```

## ⚙️ Parâmetros de Treinamento (v2.0)

### 🧠 Rede Neural
- **Entrada**: 28 características (expandido de 11)
- **Arquitetura**: DQN com 3 camadas densas
- **Neurônios Ocultos**: 512 (expandido de 256)
- **Saída**: 3 ações (frente, direita, esquerda)

### 📚 Hiperparâmetros
- **Learning Rate**: 0.001
- **Discount Factor (γ)**: 0.9
- **Epsilon Start**: 1.0 → **End**: 0.01 (exploração → exploitação)
- **Memory Size**: 10.000 experiências
- **Batch Size**: 32
- **Target Update**: 100 episódios

### 💰 Sistema de Recompensas
- **🍎 Comida**: +20 (aumentado de +10)
- **💀 Morte**: -15 (aumentado de -10)
- **🎯 Aproximar da comida**: +2
- **↩️ Afastar da comida**: -1
- **🚨 Risco de armadilha**: -8
- **🐍 Bloqueio da cauda**: -4
- **📈 Crescimento**: +0.5 por segmento

## Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Créditos

Inspirado no canal "Universo Programado" e na comunidade de IA em jogos.
