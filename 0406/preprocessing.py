import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocessing():
    """output: X_train, X_test, y_train, y_test"""
    
    
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


    # ===== 1. load data =====
    df_train = pd.read_csv("KDDTrain+.txt", header=None, names=columns)
    df_test = pd.read_csv("KDDTest+.txt", header=None, names=columns)
    
    df_train.drop('difficulty_level', axis=1, inplace=True)
    df_test.drop('difficulty_level', axis=1, inplace=True)


    # ===== 2. label mapping dictionary =====
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
    # 새로운 공격 유형 -> Others(5)로 분류
    label_num_mapping = {'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4, 'Others': 5}

    for df in [df_train, df_test]:
        df['class'] = df['class'].apply(lambda x: label_mapping.get(x, 'Others'))
        # 숫자로 변환
        df['class'] = df['class'].map(label_num_mapping)


    # ===== 3. One-hot 인코딩 및 칼럼 정렬 =====
    y_train = df_train['class']
    y_test = df_test['class']
    
    X_train = pd.get_dummies(df_train.drop('class', axis=1))
    X_test = pd.get_dummies(df_test.drop('class', axis=1))

    # test set의 칼럼을 train set의 칼럼과 일치시킴
    # train set에는 있지만 test set에는 없는 칼럼: 0으로 채움
    # 반대의 경우 제거
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # ===== 4. 정규화(Min-Max Scaling) =====
    scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ===== 5. 결과 데이터프레임 구성 =====
    df_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    df_train_final['class'] = y_train.values
    
    df_test_final = pd.DataFrame(X_test_scaled, columns=X_train.columns)
    df_test_final['class'] = y_test.values
    
    X_train = df_train_final.drop('class', axis=1)
    y_train = df_train_final['class']
    
    X_test = df_test_final.drop('class', axis=1)
    y_test = df_test_final['class']
    
    return X_train, X_test, y_train, y_test