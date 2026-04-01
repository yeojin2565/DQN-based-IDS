import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 장치: {device}")

# 1. 데이터 로드 및 전처리 (NSL-KDD)
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'class', 'difficulty_level'
]

# 데이터 로드
df_train = pd.read_csv("KDDTrain+.txt", header=None, names=columns)
df_test = pd.read_csv("KDDTest+.txt", header=None, names=columns)
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
df.drop('difficulty_level', axis=1, inplace=True)

# 5개 클래스 매핑
label_mapping = {
    'apache2': 'DoS', 'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'mailbomb': 'DoS', 
    'pod': 'DoS', 'processtable': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    'ipsweep': 'Probe', 'mscan': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'saint': 'Probe', 'satan': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'named': 'R2L', 'phf': 'R2L',
    'sendmail': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R', 'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
    'normal': 'Normal'
}
df['class'] = df['class'].replace(label_mapping)
label_num_mapping = {'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
df['class'] = df['class'].map(label_num_mapping)

# One-hot 인코딩 및 Min-Max 정규화
df = pd.get_dummies(df)
scaler = MinMaxScaler()
features = df.drop('class', axis=1)
labels = df['class']
scaled_features = scaler.fit_transform(features)
df = pd.DataFrame(scaled_features, columns=features.columns)
df['class'] = labels.values

# 학습/테스트 분할 (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('class', axis=1), df['class'], 
    test_size=0.2, random_state=42, stratify=df['class']
)
X_train_np, y_train_np = X_train.values, y_train.values

# 2. DQN 모델 및 에이전트
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
        return self.relu(self.fc3(x))

class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.state_dim, self.action_dim = input_dim, output_dim
        self.gamma = 0.001           # 최적 할인율
        self.epsilon = 0.9           # 초기 탐험율
        self.epsilon_min = 0.1       # Exploration Threshold
        self.epsilon_decay = 0.99    # Decay rate
        
        self.model = QNetwork(input_dim, output_dim).to(device)
        self.target_model = QNetwork(input_dim, output_dim).to(device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_actions_with_prob(self, states, eval_mode=False):
        states_t = torch.FloatTensor(states).to(device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(states_t)
            probs = torch.softmax(q_values, dim=1) # 확률 변환
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            max_probs = torch.max(probs, dim=1)[0].cpu().numpy()
        
        if not eval_mode and np.random.rand() <= self.epsilon:
            # 탐험: 무작위 행동 선택
            return np.random.randint(0, self.action_dim, size=len(states)), max_probs
        
        return actions, max_probs

    def train_on_batch(self, states, actions, rewards, next_states):
        self.model.train()
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        
        # 현재 Q값 예측
        current_q = self.model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # 타겟 Q값 계산: r + gamma * max Q(s', a')
            next_q = self.target_model(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q
        
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# 3. 메인 학습 루프 (전체 순회 및 확률 보상 반영)
agent = DQNAgent(X_train.shape[1], 5)
batch_size = 500

print("\n[DQL 학습 시작]")
for episode in range(250):
    perm = np.random.permutation(len(X_train_np))
    X_shuffled, y_shuffled = X_train_np[perm], y_train_np[perm]
    
    total_loss = 0
    start_idx = 0

    for t in range(100):
        states = X_shuffled[start_idx : start_idx + batch_size]
        labels = y_shuffled[start_idx : start_idx + batch_size]
        
        # 행동 결정 및 판단 확신도(Confidence) 획득
        actions, confidence = agent.get_actions_with_prob(states)
        
        # 확률 기반 보상: 확신도가 높을수록 큰 보상/패널티 
        rewards = np.where(actions == labels, 1.0 * confidence, -1.0 * (1 - confidence))
        
        # 다음 상태 S' Fetch (다음 배치 데이터)
        next_start = start_idx + batch_size
        next_states = X_shuffled[next_start : next_start + batch_size]
        if len(next_states) < batch_size: break
            
        total_loss += agent.train_on_batch(states, actions, rewards, next_states)

    # 탐험율 감소 (Epsilon Decay)
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    
    # 에피소드 종료 시 타겟 네트워크 업데이트
    agent.update_target_model()
    
    if (episode + 1) % 10 == 0:
        avg_loss = total_loss / (len(X_train_np) // batch_size)
        print(f"Episode {episode+1}/250 | Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")

# 4. 시각화 및 평가
def plot_results(agent, X_test, y_test):
    print("\n[최종 평가 및 시각화]")
    agent.model.eval()
    y_pred, _ = agent.get_actions_with_prob(X_test.values, eval_mode=True)
    classes = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    cm = confusion_matrix(y_test, y_pred)

    # Confusion Matrix 히트맵
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (DQL Model)')
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.show()

    # 클래스별 탐지 결과 바 차트
    correct, fn, fp = [], [], []
    for i in range(len(classes)):
        c = cm[i, i]
        correct.append(c)
        fn.append(np.sum(cm[i, :]) - c) # 실제 Positive인데 놓친 것
        fp.append(np.sum(cm[:, i]) - c) # 실제 Negative인데 Positive로 오해한 것
    
    x = np.arange(len(classes))
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, correct, width, label='Correct Estimation', color='#f2f0f7', edgecolor='black')
    plt.bar(x, fn, width, label='False Negative', color='#bcbddc', edgecolor='black')
    plt.bar(x + width, fp, width, label='False Positive', color='#74a9cf', edgecolor='black')
    plt.xticks(x, classes); plt.legend(); plt.ylabel('# of Samples'); plt.title('Replication of Figure 8')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

plot_results(agent, X_test, y_test)