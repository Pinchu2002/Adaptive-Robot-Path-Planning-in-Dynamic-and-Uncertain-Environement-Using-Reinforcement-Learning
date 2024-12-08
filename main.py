# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pygame
import random
import sys
import numpy as np
import json
from config import Config
from dqn_agent import Agent
import torch
from collections import deque
import time 
import matplotlib.pyplot as plt

def main():
    # Initialize Pygame
    pygame.init()

    # Load best hyperparameters from optimization
    # try:
    #     with open('best_hyperparameters_no_transformer.json', 'r') as f:
    #         best_params_no_transformer = json.load(f)
    # except FileNotFoundError:
    #     print("Best hyperparameters not found. Please run optimize.py first.")
    #     return
    
    # # Initialize configuration with best hyperparameters
    # config = Config(
    #     hidden_size=best_params_no_transformer.get('hidden_size', 128),
    #     learning_rate=best_params_no_transformer.get('learning_rate', 0.001),
    #     gamma=best_params_no_transformer.get('gamma', 0.99),
    #     epsilon_decay=best_params_no_transformer.get('epsilon_decay', 0.995),
    #     use_transformer=best_params_no_transformer.get('use_transformer', False),
    #     transformer_nhead=best_params_no_transformer.get('transformer_nhead', 4),
    #     transformer_num_layers=best_params_no_transformer.get('transformer_num_layers', 2)
    # )

    # # Load best hyperparameters with Transformer
    try:
        with open('best_hyperparamters_with_transformer_1
        
        .json', 'r') as f:
            best_params_with_transformer = json.load(f)
    except:
        print("Best hyperparameters not found. Please run optimize.py first.")
        return
    

    # Initialize configuration with best hyperparameters with Transformer
    config = Config(
        hidden_size=best_params_with_transformer.get('hidden_size', 128),
        learning_rate=best_params_with_transformer.get('learning_rate', 0.001),
        gamma=best_params_with_transformer.get('gamma', 0.99),
        epsilon_decay=best_params_with_transformer.get('epsilon_decay', 0.995),
        use_transformer=best_params_with_transformer.get('use_transformer', True),
        transformer_nhead=best_params_with_transformer.get('transformer_nhead', 4),
        transformer_num_layers=best_params_with_transformer.get('transformer_num_layers', 2)
    )
    
    # Initialize the screen
    screen = pygame.display.set_mode((config.window_size, config.window_size))
    pygame.display.set_caption("Robot Path Planning")

    # Initialize font for numbering path cells
    font = pygame.font.Font(None, 24)  # Default font with size 24

    # Grid representation
    grid = [[0 for _ in range(config.grid_size)] for _ in range(config.grid_size)]
    start = None
    destination = None

    # Initialize agent with best configuration
    agent = Agent(config)
    
    # Initialize dynamic obstacles, etc.
    dynamic_obstacles = [(5, 5), (10, 10), (3, 3), (8, 8), (16, 16), (18, 18), (7, 7), (20, 20), (24, 24), (28, 28)]  # Initial positions of dynamic obstacles
    dynamic_directions = [(1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)]  # Directions in which the dynamic obstacles move
    initial_dynamic_obstacles = dynamic_obstacles

    # Experience Replay and visited_cells
    memory = deque(maxlen=config.memory_capacity)
    visited_cells = np.zeros((config.grid_size, config.grid_size))  # Track cell visits to increase penalty

    # Function Definitions
    def draw_grid():
        """Draw the grid lines."""
        for x in range(0, config.window_size, config.cell_size):
            for y in range(0, config.window_size, config.cell_size):
                rect = pygame.Rect(x, y, config.cell_size, config.cell_size)
                pygame.draw.rect(screen, (200, 200, 200), rect, 1)

    def draw_cells():
        """Draw cells based on grid data."""
        for i in range(config.grid_size):
            for j in range(config.grid_size):
                color = config.bg_color
                if grid[i][j] == 1:
                    color = config.obstacle_color
                elif (i, j) == start:
                    color = config.start_color
                elif (i, j) == destination:
                    color = config.dest_color
                pygame.draw.rect(screen, color,
                                 (j * config.cell_size, i * config.cell_size, config.cell_size, config.cell_size))
    
        # Draw dynamic obstacles
        for obstacle in dynamic_obstacles:
            pygame.draw.rect(screen, config.dynamic_obstacle_color,
                             (obstacle[1] * config.cell_size, obstacle[0] * config.cell_size, config.cell_size, config.cell_size))

    def highlight_shortest_path(parent, current_pos):
        """Highlight the shortest path in yellow with numbering and inner padding."""
        step_count = 1
        padding = 4  # Padding inside each cell for visualization
        while current_pos != start:
            pygame.draw.rect(
                screen,
                config.path_color,
                (current_pos[1] * config.cell_size + padding, current_pos[0] * config.cell_size + padding,
                 config.cell_size - 2 * padding, config.cell_size - 2 * padding),
            )
            # Render the step number
            text_surface = font.render(str(step_count), True, config.text_color)
            text_rect = text_surface.get_rect(center=(current_pos[1] * config.cell_size + config.cell_size // 2,
                                                      current_pos[0] * config.cell_size + config.cell_size // 2))
            screen.blit(text_surface, text_rect)

            current_pos = parent.get(current_pos, start)
            step_count += 1
        pygame.display.flip()

    def move_robot(state):
        """Move the robot using DQN."""
        action = agent.choose_action(state)
        dx, dy = config.actions[action]
        new_x, new_y = state[0] + dx, state[1] + dy

        # Check boundaries and obstacles
        if 0 <= new_x < config.grid_size and 0 <= new_y < config.grid_size and grid[new_x][new_y] != 1 and (new_x, new_y) not in dynamic_obstacles:
            visited_cells[new_x][new_y] += 1
            penalty = visited_cells[new_x][new_y] * -0.1  # Increase penalty for revisited cells
            return (new_x, new_y), -1 + penalty  # Penalty for moving, with extra for revisits
        visited_cells[state[0]][state[1]] += 2  # Additional penalty for getting stuck in the same cell
        return state, -100  # Penalty for hitting obstacle or invalid move

    def move_dynamic_obstacles():
        """Move dynamic obstacles in their respective directions."""
        for index, (obstacle, direction) in enumerate(zip(dynamic_obstacles, dynamic_directions)):
            new_x = obstacle[0] + direction[0]
            new_y = obstacle[1] + direction[1]

            # Check boundaries and reverse direction if needed
            if not (0 <= new_x < config.grid_size and 0 <= new_y < config.grid_size) or grid[new_x][new_y] == 1:
                dynamic_directions[index] = (-direction[0], -direction[1])  # Reverse direction
            else:
                dynamic_obstacles[index] = (new_x, new_y)

    episode_times = []

    # Main loop
    running = True
    robot_pos = None
    episode = 0
    parent = {}  # Dictionary to store parent of each cell
    best_path = None
    best_steps = float('inf')
    
    # Optionally, get the number of sets from the user
    try:
        num_sets = int(input("Enter the number of sets of 10 episodes (e.g., 1, 2, 3): "))
    except ValueError:
        num_sets = 1  # Default to 1 set if invalid input
    num_episodes = num_sets * 10

    while running and episode < num_episodes:
        screen.fill(config.bg_color)
        draw_cells()
        draw_grid()
        move_dynamic_obstacles()  # Move dynamic obstacles each iteration

        if episode == 0 or (episode % 10 == 0 and timestamp_start is None):
            timestamp_start = time.time() 

        # Visualization of robot movement
        if robot_pos:
            pygame.draw.rect(
                screen,
                config.robot_color,
                (robot_pos[1] * config.cell_size, robot_pos[0] * config.cell_size, config.cell_size, config.cell_size),
            )
            pygame.display.flip()

            next_pos, reward = move_robot(robot_pos)
            dx = next_pos[0] - robot_pos[0]
            dy = next_pos[1] - robot_pos[1]

            # Ensure valid action index
            if (dx, dy) in config.actions:
                action_index = config.actions.index((dx, dy))
            else:
                action_index = 0  # Default action for no movement or invalid move

            done = next_pos == destination

            # Store transition and track the parent
            agent.store_transition((list(robot_pos), action_index, reward, list(next_pos), done))
            
            # Track parent only if moving to a new position
            if next_pos not in parent:
                parent[next_pos] = robot_pos  # Set parent for backtracking

            if done:
                dynamic_obstacles = initial_dynamic_obstacles.copy()
                print(f"Destination Reached in Episode {episode}!")
                steps = len(parent)
                if steps < best_steps:
                    best_steps = steps
                    best_path = parent.copy()
                if (episode + 1) % 10 == 0:
                    end_time = time.time()  # End time for this episode
                    episode_time = end_time - timestamp_start  # Calculate the time taken for this episode set
                    episode_times.append(episode_time)
                    print("Episode Time : ",episode_time)
                    highlight_shortest_path(best_path, destination)  # Highlight the best path after each set of 10 episodes
                    print("Press Enter to continue or exit...")
                    while True:
                        event = pygame.event.wait()
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                            break
                # Reset for next episode (except grid, start, and destination)
                robot_pos = start
                parent = {}
                episode += 1
                agent.update_target()
                print(f"Starting Episode {episode}...")

                if episode % 10 == 0:
                    # time_taken_for_set = sum(episode_times[-10:])  # Sum the last 10 episode times
                    print(f"Time taken for Episodes {episode-9} to {episode}: {episode_time:.2f} seconds")
                    timestamp_start = None  # Reset timestamp start for the next set


            else:
                robot_pos = next_pos

            agent.train()

        pygame.display.flip()
        pygame.time.wait(10)  # Delay to control the speed of robot movement

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                grid_x, grid_y = y // config.cell_size, x // config.cell_size

                if event.button == 1:  # Left-click
                    if not start:
                        start = (grid_x, grid_y)
                    elif not destination:
                        destination = (grid_x, grid_y)
                    else:
                        grid[grid_x][grid_y] = 1

                elif event.button == 3:  # Right-click
                    if grid[grid_x][grid_y] == 1:
                        grid[grid_x][grid_y] = 0

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and destination:  # Start learning
                    robot_pos = start
                    episode += 1
                    print(f"Starting Episode {episode}...")


    # print()
    plt.plot(range(1, len(episode_times) + 1), episode_times)
    plt.xlabel('Set Number')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Set for Robot Path Finding')
    plt.grid(True)
    plt.show()
    print(episode_times)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
        main()



