#!/usr/bin/env python3
"""
Teste direto da interface completa
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.interface.game_interface import GameInterface

def test_interface():
    print("Testando Interface Completa")
    print("Instruções:")
    print("- Quando a janela abrir, pressione 'M' para modo manual")
    print("- Use as setas do teclado para mover")
    print("- Pressione ESC para sair")
    
    interface = GameInterface()
    interface.run()

if __name__ == "__main__":
    test_interface()
