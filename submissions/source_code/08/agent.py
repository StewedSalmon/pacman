"""
python arena.py --seek group_A --hide example_student --pacman-obs-radius 5 --ghost-obs-radius 5 --pacman-speed 2 --delay 0.1
python arena.py --seek example_student --hide group_A --pacman-obs-radius 5 --ghost-obs-radius 5 --pacman-speed 2 --delay 0.1
python arena.py --seek group_A --hide group_A --pacman-obs-radius 5 --ghost-obs-radius 5 --pacman-speed 2 --delay 0.1
"""

from pacman import PacmanAgent
from ghost import GhostAgent
