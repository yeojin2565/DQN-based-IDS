"""
NSL-KDD IDS — DQN 기반 침입 탐지 시스템
- preprocess.py (nslkdd_loader.py) 에서 정규화된 environment 행렬 로드
- 각 샘플을 state로 받아 normal(0) / attack(1) 이진 분류
- Hyperparameter: Table 3 기준
    num-episode   : 200
    num-iteration : 100 (에피소드당 최대 스텝)
    hidden_layers : 2
    num_units     : 2 × 100
    Epsilon ε     : 0.9
    Decay rate    : 0.99 (에피소드마다 감소)
    Gamma γ       : 0.001
    Batch-size    : 500
    Weight init   : Normal distribution
    Activation    : ReLU
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import preprocess as pc   # nslkdd_loader.py 를 preprocess.py 로 사용

# ──────────────────────────────────────────────
# 디바이스
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# Hyperparameters (Table 3)
# ──────────────────────────────────────────────
NUM_EPISODES   = 200        # num-episode
NUM_ITERATIONS = 100        # num-iteration (에피소드 당 최대 스텝 수)
HIDDEN_LAYERS  = 2          # hidden_layers
NUM_UNITS      = 100        # num_units (레이어당 뉴런 수, 2 × 100)
EPSILON        = 0.9        # 초기 ε
DECAY_RATE     = 0.99       # decay rate (에피소드마다 ε 감소)
EPSILON_MIN    = 0.01       # ε 하한
GAMMA          = 0.001      # discount factor γ
BATCH_SIZE     = 500        # batch-size
LR             = 1e-3       # learning rate (Adam)
BUFFER_CAP     = 10000      # Replay Buffer 용량
TARGET_UPDATE  = 10         # target network 동기화 주기 (에피소드 단위)

ACTION_DIM = 2              # 0: normal, 1: attack


# ══════════════════════════════════════════════
# 1. IDS 커스텀 환경
# ══════════════════════════════════════════════
class IDSEnvironment:
    """
    NSL-KDD 데이터를 Gym-like 인터페이스로 감싸는 커스텀 환경.

    - state  : 정규화된 피처 벡터 (X_norm[i])
    - action : 0 = normal 예측, 1 = attack 예측
    - reward : 정답이면 +1.0, 오답이면 -1.0
    - done   : NUM_ITERATIONS 스텝 또는 데이터 소진 시 True
    """

    def __init__(self, X: np.ndarray, y_binary: np.ndarray):
        self.X        = X.astype(np.float32)
        self.y        = y_binary                # 0=normal, 1=attack
        self.n        = len(X)
        self.state_dim = X.shape[1]

        self._indices  = np.arange(self.n)
        self._ptr      = 0
        self._step_cnt = 0

    # ── 에피소드 시작 ──────────────────────────
    def reset(self) -> np.ndarray:
        np.random.shuffle(self._indices)
        self._ptr      = 0
        self._step_cnt = 0
        return self.X[self._indices[self._ptr]]

    # ── 한 스텝 ───────────────────────────────
    def step(self, action: int):
        true_label = int(self.y[self._indices[self._ptr]])
        reward     = 1.0 if action == true_label else -1.0

        self._ptr      += 1
        self._step_cnt += 1
        done = (self._step_cnt >= NUM_ITERATIONS) or (self._ptr >= self.n)

        if not done:
            next_state = self.X[self._indices[self._ptr]]
        else:
            next_state = np.zeros(self.state_dim, dtype=np.float32)

        return next_state, reward, done, true_label


# ══════════════════════════════════════════════
# 2. DQN 네트워크
#    구조: state_dim → 100 → 100 → 2
#    Weight Init: Normal distribution
#    Activation: ReLU
# ══════════════════════════════════════════════
class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = ACTION_DIM):
        super(DQN, self).__init__()

        # 2개의 은닉층, 각 100 유닛
        layers = []
        in_dim = state_dim
        for _ in range(HIDDEN_LAYERS):
            layers.append(nn.Linear(in_dim, NUM_UNITS))
            layers.append(nn.ReLU())
            in_dim = NUM_UNITS
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Normal distribution 가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════
# 3. Replay Buffer
# ══════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_CAP):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: tuple):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════
# 4. 데이터 로드 (preprocess.py)
# ══════════════════════════════════════════════
print("=" * 55)
print(" NSL-KDD 데이터 로드 중 (preprocess.py)...")
print("=" * 55)

ARFF_PATH = "KDDTrain+.arff"   # ← 실제 경로로 수정
result    = pc.load_nslkdd(ARFF_PATH)

X_norm    = result["X_norm"].astype(np.float32)   # 정규화된 피처 행렬
y_labels  = result["y_labels"]                     # 원본 문자열 레이블

# normal=0, 그 외(attack)=1 이진 변환
y_binary  = np.where(y_labels == "normal", 0, 1).astype(np.int64)

print(f"\n  피처 차원    : {X_norm.shape[1]}")
print(f"  총 샘플 수   : {X_norm.shape[0]:,}")
print(f"  normal 수   : {(y_binary==0).sum():,}")
print(f"  attack 수   : {(y_binary==1).sum():,}\n")

STATE_DIM = X_norm.shape[1]

# ── 훈련 / 평가 분리 (80:20) ──────────────────
n_train = int(len(X_norm) * 0.8)
X_train, X_test   = X_norm[:n_train],   X_norm[n_train:]
y_train, y_test   = y_binary[:n_train], y_binary[n_train:]


# ══════════════════════════════════════════════
# 5. 모델 / 옵티마이저 초기화
# ══════════════════════════════════════════════
env          = IDSEnvironment(X_train, y_train)
policy_net   = DQN(STATE_DIM).to(device)
target_net   = DQN(STATE_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer    = optim.Adam(policy_net.parameters(), lr=LR)
replay_buf   = ReplayBuffer(BUFFER_CAP)

epsilon      = EPSILON
reward_list  = []
acc_list     = []


# ══════════════════════════════════════════════
# 6. 학습 루프
# ══════════════════════════════════════════════
print("=" * 55)
print(f" DQN 학습 시작 | episodes={NUM_EPISODES} | iter/ep={NUM_ITERATIONS}")
print("=" * 55)

for episode in range(1, NUM_EPISODES + 1):

    state = env.reset()
    state = torch.FloatTensor(state).to(device)

    total_reward  = 0.0
    correct       = 0
    total_steps   = 0

    for _ in range(NUM_ITERATIONS):

        # ε-greedy 행동 선택
        if np.random.rand() < epsilon:
            action = np.random.randint(ACTION_DIM)
        else:
            with torch.no_grad():
                action = policy_net(state).argmax().item()

        next_state, reward, done, true_label = env.step(action)

        next_state_t = torch.FloatTensor(next_state).to(device)
        replay_buf.push((state, action, reward, next_state_t, done))

        state         = next_state_t
        total_reward += reward
        correct      += int(action == true_label)
        total_steps  += 1

        # ── 미니배치 업데이트 ───────────────────
        if len(replay_buf) >= BATCH_SIZE:
            transitions = replay_buf.sample(BATCH_SIZE)
            b_s, b_a, b_r, b_ns, b_d = zip(*transitions)

            b_s  = torch.stack(b_s).to(device)
            b_a  = torch.LongTensor(b_a).unsqueeze(1).to(device)
            b_r  = torch.FloatTensor(b_r).unsqueeze(1).to(device)
            b_ns = torch.stack(b_ns).to(device)
            b_d  = torch.FloatTensor(b_d).unsqueeze(1).to(device)

            # Q(s, a)
            q_values = policy_net(b_s).gather(1, b_a)

            # 타깃: r + γ * max Q'(s', a') * (1 - done)
            with torch.no_grad():
                next_q = target_net(b_ns).max(1)[0].unsqueeze(1)
            target = b_r + GAMMA * next_q * (1 - b_d)

            loss = F.mse_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # ε 감소 (decay rate 적용)
    epsilon = max(EPSILON_MIN, epsilon * DECAY_RATE)

    # target network 동기화
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    episode_acc = correct / total_steps if total_steps > 0 else 0.0
    reward_list.append(total_reward)
    acc_list.append(episode_acc)

    if episode % 20 == 0 or episode == 1:
        print(f"  Episode {episode:>3}/{NUM_EPISODES} | "
              f"Reward: {total_reward:>7.1f} | "
              f"Acc: {episode_acc:.4f} | "
              f"ε: {epsilon:.4f}")


# ══════════════════════════════════════════════
# 7. 평가 (테스트셋)
# ══════════════════════════════════════════════
print("\n" + "=" * 55)
print(" 테스트셋 평가 중...")
print("=" * 55)

policy_net.eval()
X_test_t    = torch.FloatTensor(X_test).to(device)

with torch.no_grad():
    q_vals      = policy_net(X_test_t)
    y_pred      = q_vals.argmax(dim=1).cpu().numpy()

print("\n[Classification Report]")
print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

cm = confusion_matrix(y_test, y_pred)
print("[Confusion Matrix]")
print(f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
print(f"  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
test_acc = (y_pred == y_test).mean()
print(f"\n  Test Accuracy : {test_acc:.4f}")


# ══════════════════════════════════════════════
# 8. 학습 그래프
# ══════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# ── 리워드 ──────────────────────────────────
ax1.plot(reward_list, alpha=0.4, color="steelblue", label="Episode Reward")
window = 20
if len(reward_list) >= window:
    ma = [np.mean(reward_list[i-window:i]) for i in range(window, len(reward_list)+1)]
    ax1.plot(range(window-1, len(reward_list)), ma,
             color="crimson", linewidth=2, label=f"{window}-ep Moving Avg")
ax1.set_ylabel("Total Reward")
ax1.set_title("DQN IDS Training — NSL-KDD (normal vs attack)")
ax1.legend()
ax1.grid(alpha=0.3)

# ── 정확도 ──────────────────────────────────
ax2.plot(acc_list, alpha=0.4, color="darkorange", label="Episode Accuracy")
if len(acc_list) >= window:
    ma_acc = [np.mean(acc_list[i-window:i]) for i in range(window, len(acc_list)+1)]
    ax2.plot(range(window-1, len(acc_list)), ma_acc,
             color="darkgreen", linewidth=2, label=f"{window}-ep Moving Avg")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1.05)
ax2.axhline(test_acc, color="navy", linestyle="--", linewidth=1.5,
            label=f"Test Acc = {test_acc:.4f}")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ids_dqn_training.png", dpi=150)
plt.show()
print("\n그래프 저장: ids_dqn_training.png")