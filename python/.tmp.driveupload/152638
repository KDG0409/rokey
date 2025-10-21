# Xor 연산 학습

import sklearn as sk
from sklearn import svm
from sklearn import metrics
import pandas as pd

# 데이터 분할
xor_data=[
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]    
]
xor_df = pd.DataFrame(xor_data)
data = xor_df.loc[:,0:1]
label = xor_df.loc[:,2]

# 분할된 데이터로 학습
clf = svm.SVC()
clf.fit(data,label)

# 데이터 예측값 생성
pre = clf.predict(data)
print(pre)

#정답과 예측값 정답률 확인
ac_score = metrics.accuracy_score(label,pre)
print(f"정답률:{ac_score}")