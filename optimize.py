import optuna
from config import Config
from dqn_agent import Agent
import torch
import json
import gc

def objective(trial):
    # Suggest hyperparameters using non-deprecated methods
    use_transformer = trial.suggest_categorical('use_transformer', [True, False])
    
    if use_transformer:
        # Fixed set of nhead values
        transformer_nhead = trial.suggest_categorical('transformer_nhead', [2, 4, 8])
        
        # Determine the multiplier to ensure hidden_size is divisible by transformer_nhead
        hidden_multiplier = trial.suggest_int('hidden_multiplier', 
                                               min_multiplier := 32 // transformer_nhead, 
                                               max_multiplier := 256 // transformer_nhead)
        
        hidden_size = transformer_nhead * hidden_multiplier
        
        transformer_num_layers = trial.suggest_int('transformer_num_layers', 1, 4)
    else:
        # When not using transformer, sample hidden_size normally
        hidden_size = trial.suggest_int('hidden_size', 64, 256)
        # Assign default transformer parameters
        transformer_nhead = 4
        transformer_num_layers = 2
    
    # Initialize configuration with suggested hyperparameters
    config = Config(
        hidden_size=hidden_size,
        learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        gamma=trial.suggest_float('gamma', 0.90, 0.999),
        epsilon_decay=trial.suggest_float('epsilon_decay', 0.90, 0.999),
        use_transformer=use_transformer,
        transformer_nhead=transformer_nhead,
        transformer_num_layers=transformer_num_layers
    )
    
    print(f"Using device: {config.device}")  # For verification
    
    try:
        # Initialize agent
        agent = Agent(config)
    except AssertionError as e:
        # If hidden_size is not divisible by nhead, prune the trial
        raise optuna.exceptions.TrialPruned(f"AssertionError: {e}")
    
    # Initialize environment
    grid = [[0 for _ in range(config.grid_size)] for _ in range(config.grid_size)]
    start = (0, 0)
    destination = (config.grid_size - 1, config.grid_size - 1)
    robot_pos = start
    episode = 0
    num_episodes = 10  # Use fewer episodes for faster optimization
    
    total_steps = 0
    
    while episode < num_episodes:
        # Simple environment simulation without Pygame for faster training
        action = agent.choose_action(robot_pos)
        dx, dy = config.actions[action]
        new_x, new_y = robot_pos[0] + dx, robot_pos[1] + dy

        # Check boundaries and obstacles
        if 0 <= new_x < config.grid_size and 0 <= new_y < config.grid_size and grid[new_x][new_y] != 1:
            next_pos = (new_x, new_y)
            reward = -1  # Simple reward
        else:
            next_pos = robot_pos
            reward = -100  # Penalty for invalid move

        done = next_pos == destination
        agent.store_transition((list(robot_pos), action, reward, list(next_pos), done))
        agent.train()

        if done:
            episode += 1
            robot_pos = start
        else:
            robot_pos = next_pos
            total_steps += 1

    # Clean up to free GPU memory
    del agent
    del config
    torch.cuda.empty_cache()
    gc.collect()

    return total_steps  # Optuna will try to minimize this

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

    print("Best hyperparameters: ", study.best_params)
    
    # Save best hyperparameters to a JSON file
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
