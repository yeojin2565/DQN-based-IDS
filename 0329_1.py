import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# 컬럼 정의
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

# 데이터 로드 및 합치기
df_train = pd.read_csv("KDDTrain+.txt", header=None, names=columns)
df_test = pd.read_csv("KDDTest+.txt", header=None, names=columns)
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# 불필요한 컬럼 삭제 (difficulty_level)
df.drop('difficulty_level', axis=1, inplace=True)

label_mapping = {
    # DoS
    'apache2': 'DoS', 'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS',
    'mailbomb': 'DoS', 'pod': 'DoS', 'processtable': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',

    # Probe
    'ipsweep': 'Probe', 'mscan': 'Probe', 'nmap': 'Probe',
    'portsweep': 'Probe', 'saint': 'Probe', 'satan': 'Probe',

    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L',
    'imap': 'R2L', 'multihop': 'R2L', 'named': 'R2L', 'phf': 'R2L',
    'sendmail': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L',

    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'httptunnel': 'U2R',
    'ps': 'U2R', 'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',

    # Normal
    'normal': 'Normal'
}

df['class'] = df['class'].replace(label_mapping)

# Label Encoding (문자열 라벨 -> 숫자)
label_num_mapping = {
    'Normal': 0,
    'DoS': 1,
    'Probe': 2,
    'R2L': 3,
    'U2R': 4
}
df['class'] = df['class'].map(label_num_mapping)

# One-Hot Encoding (범주형 데이터 -> 이진 벡터)
# protocol_type, service, flag 열
df = pd.get_dummies(df)

# Min-Max Normalization (수치 데이터 0~1 스케일링)
# 수치형 컬럼
scaler = MinMaxScaler()

# 인코딩된 컬럼들을 포함하여 모든 피처(X)를 정규화
features = df.drop('class', axis=1)
labels = df['class']

# 스케일링 적용
scaled_features = scaler.fit_transform(features)

# 다시 데이터프레임으로 변환
df = pd.DataFrame(scaled_features, columns=features.columns)
df['class'] = labels.values  # 라벨 다시 합치기

# Shuffling (데이터 무작위 섞기)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 최종 결과 확인
print("전처리 완료!")
print(f"데이터 크기: {df.shape}")
print(df.head())


# 특성(X)과 라벨(y) 분리
# 'labels' 컬럼은 정답지이므로 y에 넣고, 나머지는 X에 넣습니다.
X = df.drop('class', axis=1)
y = df['class']

# 데이터를 Train과 Test 세트로 분할 (80% : 20%)
# test_size=0.2 : 테스트 데이터를 20%로 설정
# random_state=42 : 실행할 때마다 동일한 결과를 얻기 위해 난수 고정
# stratify=y : 원본 데이터의 라벨 비율(Normal, DoS 등)을 유지하며 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 결과 확인
print(f"전체 데이터 개수: {len(df)}")
print(f"학습용 데이터 (X_train): {X_train.shape}, (y_train): {y_train.shape}")
print(f"테스트용 데이터 (X_test): {X_test.shape}, (y_test): {y_test.shape}")

# 라벨 비율이 잘 유지되었는지 확인
print("\n[학습 데이터 라벨 분포]")
print(y_train.value_counts(normalize=True))


# 1. DQN 모델 정의
class DQN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes) # Q-value 출력
        )

    def forward(self, x):
        return self.net(x)

# 2. DQL 에이전트 클래스
class DQLAgent:
    def __init__(self, input_dim, num_classes):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 및 최적화 설정
        self.model = DQN(input_dim, num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # 하이퍼파라미터
        self.gamma = 0.001        # 할인 계수
        self.epsilon = 0.9        # 초기 탐사 확률
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.batch_size = 500     # 배치 사이즈

    def choose_action(self, state_tensor):
        if np.random.rand() <= self.epsilon:
            return torch.randint(0, self.num_classes, (state_tensor.shape[0],)).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            return torch.argmax(q_values, dim=1)

    def train(self, X_train, y_train, episodes=250, iterations=100):
        # 데이터를 PyTorch Tensor로 변환 (KeyError 방지)
        X_train_t = torch.FloatTensor(X_train.values).to(self.device)
        y_train_t = torch.LongTensor(y_train.values).to(self.device)
        num_samples = X_train_t.shape[0]

        for ep in range(episodes):
            for it in range(iterations):
                # 500개 무작위 인덱스 추출 
                indices = torch.randperm(num_samples - 1)[:self.batch_size]
                
                states = X_train_t[indices]
                labels = y_train_t[indices]
                next_states = X_train_t[indices + 1]

                # 행동 선택 및 보상 계산
                actions = self.choose_action(states)
                rewards = torch.where(actions == labels, 1.0, -1.0).to(self.device)

                # Target Q 계산: R + gamma * max Q(s', a')
                with torch.no_grad():
                    next_q = self.model(next_states)
                    max_next_q = torch.max(next_q, dim=1)[0]
                    target_q_values = rewards + self.gamma * max_next_q

                # 현재 Q 예측 및 손실 계산
                current_q = self.model(states)
                current_q_action = current_q.gather(1, actions.unsqueeze(1)).squeeze()
                
                loss = self.criterion(current_q_action, target_q_values)

                # 역전파 및 최적화 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Epsilon 감쇠
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            if (ep + 1) % 10 == 0:
                print(f"에피소드 {ep+1}/{episodes} | 탐험률: {self.epsilon:.4f} | 손실: {loss.item():.4f}")

# 3. 모델 실행 및 테스트
agent = DQLAgent(input_dim=122, num_classes=5)

print("PyTorch로 학습을 시작합니다...")
agent.train(X_train, y_train)

# 테스트 평가
print("\n테스트 데이터 평가 중...")
X_test_t = torch.FloatTensor(X_test.values).to(agent.device)
with torch.no_grad():
    test_q = agent.model(X_test_t)
    test_preds = torch.argmax(test_q, dim=1).cpu().numpy()

final_acc = np.mean(test_preds == y_test.values)
print(f"최종 테스트 정확도: {final_acc * 100:.2f}%")