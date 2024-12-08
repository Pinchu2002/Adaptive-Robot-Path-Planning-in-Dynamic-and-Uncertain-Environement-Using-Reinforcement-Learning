
# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=128,
        use_transformer=False,
        nhead=4,
        num_layers=2,
        device='cpu'  # Added device parameter
    ):
        super(DQN, self).__init__()
        self.use_transformer = use_transformer
        self.device = device
        if self.use_transformer:
            logger.info("Initializing Transformer layers with batch_first=True.")
            # Input projection to hidden_size
            self.input_proj = nn.Linear(input_dim, hidden_size)
            
            # Ensure hidden_size is divisible by nhead
            assert hidden_size % nhead == 0, "hidden_size must be divisible by nhead"
            
            # Initialize Transformer Encoder with batch_first=True
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=nhead,
                    batch_first=True  # Important to set batch_first=True
                ),
                num_layers=num_layers
            )
            self.fc = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dim)
            )
        else:
            logger.info("Initializing fully connected layers without Transformer.")
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dim)
            )
        # Move model to device
        self.to(self.device)

    def forward(self, x):
        if self.use_transformer:
            x = self.input_proj(x)  # Project input to hidden_size
            # Transformer expects (batch_size, sequence_length, d_model) with batch_first=True
            # Since sequence_length=1, we can add a dimension
            x = x.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
            x = self.transformer(x)  # Shape: (batch_size, 1, hidden_size)
            x = x.squeeze(1)  # Shape: (batch_size, hidden_size)
        return self.fc(x)

class Agent:
    def __init__(self, config):
        self.config = config
        self.device = config.device  # Get device from config
        self.eval_net = DQN(
            input_dim=2,
            output_dim=len(config.actions),
            hidden_size=config.hidden_size,
            use_transformer=config.use_transformer,
            nhead=config.transformer_nhead,
            num_layers=config.transformer_num_layers,
            device=self.device  # Pass device to DQN
        )
        self.target_net = DQN(
            input_dim=2,
            output_dim=len(config.actions),
            hidden_size=config.hidden_size,
            use_transformer=config.use_transformer,
            nhead=config.transformer_nhead,
            num_layers=config.transformer_num_layers,
            device=self.device  # Pass device to DQN
        )
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.epsilon = config.epsilon
        self.memory = deque(maxlen=config.memory_capacity)
        
        # Log device information
        logger.info(f"Eval Net is on device: {next(self.eval_net.parameters()).device}")
        logger.info(f"Target Net is on device: {next(self.target_net.parameters()).device}")

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.config.actions) - 1)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)  # Move to device
        with torch.no_grad():
            q_values = self.eval_net(state)
        return torch.argmax(q_values).item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def train(self):
        if len(self.memory) < self.config.batch_size:
            return

        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors and move to device
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # Current Q values
        q_eval = self.eval_net(states).gather(1, actions).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0]
        
        # Compute target Q values
        q_target = rewards + self.config.gamma * q_next * (1 - dones)

        # Compute loss
        loss = self.loss_fn(q_eval, q_target)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
