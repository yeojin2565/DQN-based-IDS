import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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