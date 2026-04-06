import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from preprocessing import load_and_preprocessing
from model import DQNAgent

# 학습 / 테스트 분할
X_train, X_test, y_train, y_test= load_and_preprocessing()
X_train_np, y_train_np = X_train.values, y_train.values

# ===== 메인 학습 루프 =====
agent = DQNAgent(X_train.shape[1], 6) # others 추가

print("\n[학습 시작]")
for episode in range(250):
    perm = np.random.permutation(len(X_train_np))
    X_shuffled, y_shuffled = X_train_np[perm], y_train_np[perm]
    total_loss, start_idx = 0, 0
    
    for t in range(100):
        states = X_shuffled[start_idx : start_idx + 500]
        labels = y_shuffled[start_idx : start_idx + 500]
        
        actions = agent.get_actions(states)
        rewards = np.where(actions == labels, 1.0, -0.01)
        
        next_start = start_idx + 500
        if next_start + 500 <= len(X_shuffled):
            next_states = X_shuffled[next_start : next_start + 500]
        else:
            next_states = states
            
        total_loss += agent.train_on_batch(states, actions, rewards, next_states)
        start_idx += 500

    agent.epsilon *= agent.epsilon_decay
    agent.update_target_model()
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}/250 | Avg Loss: {total_loss/100:.4f} | Epsilon: {agent.epsilon:.3f}")
        

# ===== 시각화 및 평가 =====
def plot_final_results(agent, X_test, y_test):
    agent.model.eval()
    y_pred = agent.get_actions(X_test.values, eval_mode=True)
    classes = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R', 'Others'] # Others 추가
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Heatmap)')
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.show()

    correct, fn, fp = [], [], []
    for i in range(len(classes)):
        c = cm[i, i]
        correct.append(c)
        fn.append(np.sum(cm[i, :]) - c)
        fp.append(np.sum(cm[:, i]) - c)
    
    x = np.arange(len(classes))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, correct, width, label='Correct Estimation', color='#f2f0f7', edgecolor='black')
    plt.bar(x, fn, width, label='False Negative', color='#bcbddc', edgecolor='black')
    plt.bar(x + width, fp, width, label='False Positive', color='#74a9cf', edgecolor='black')
    plt.xticks(x, classes); plt.legend(); plt.title('Classification Performance (Correct vs Error)')
    plt.show()

    print("\n[최종 성능 보고서]\n", classification_report(y_test, y_pred, target_names=classes))

plot_final_results(agent, X_test, y_test)