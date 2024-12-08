# config.py
from dataclasses import dataclass
import torch

@dataclass
class Config:
    window_size: int = 600
    grid_size: int = 20
    cell_size: int = 30  # window_size // grid_size
    start_color: tuple = (0, 255, 0)
    dest_color: tuple = (255, 0, 0)
    obstacle_color: tuple = (0, 0, 0)
    dynamic_obstacle_color: tuple = (255, 165, 0)
    bg_color: tuple = (135, 206, 235)
    robot_color: tuple = (0, 0, 255)
    path_color: tuple = (255, 255, 0)
    text_color: tuple = (0, 0, 0)
    
    actions: list = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, 1), (1, -1), (-1, 1))
    gamma: float = 0.99
    learning_rate: float = 0.001
    batch_size: int = 64
    memory_capacity: int = 10000
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.1
    target_update: int = 5
    hidden_size: int = 128  # Example additional hyperparameter
    
    # Transformer-specific hyperparameters
    use_transformer: bool = False
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    
    # Device configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'