import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 1. DQN 모델 =====
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # 출력층 ReLU --> 04.06 제거
    

# ===== 2. DQN 에이전트 =====
class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.state_dim, self.action_dim = input_dim, output_dim
        self.device = device
        print(f"현재 사용 중인 장치: {self.device}")
        
        # Hyperparameters
        self.gamma = 0.001
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        self.batch_size = 500
        
        self.model = QNetwork(input_dim, output_dim).to(self.device)
        self.target_model = QNetwork(input_dim, output_dim).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def get_actions(self, states, eval_mode=False):
        if not eval_mode and np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_dim, size=len(states))
        
        states_t = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(states_t), dim=1).cpu().numpy()
        
    def train_on_batch(self, states, actions, rewards, next_states):
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        
        current_q = self.model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q = rewards_t + self.gamma * self.target_model(next_states_t).max(1)[0]
            
        # 논문의 커스텀 손실 함수 구현
        d_pred = current_q.detach() + 1e-6
        loss = torch.mean(((current_q / d_pred) - target_q)**2)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()