"""
python arena.py --seek group_8 --hide example_student --pacman-obs-radius 5 --ghost-obs-radius 5 --pacman-speed 2 --delay 0.1
python arena.py --seek example_student --hide group_8 --pacman-obs-radius 5 --ghost-obs-radius 5 --pacman-speed 2 --delay 0.1
python arena.py --seek group_8 --hide group_8 --pacman-obs-radius 5 --ghost-obs-radius 5 --pacman-speed 2 --delay 0.1
"""

from pacman import PacmanAgent
from ghost import GhostAgent
