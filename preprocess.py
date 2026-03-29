# NSL-KDD 데이터셋 가져오기
# min-max 정규화
# label one-hot encoding
# train / test 분리

# ===== environment =====
import pandas as pd
from scipy.io import arff

# 1. arff 읽기
data, meta = arff.loadarff("KDDTrain+.arff")