# 붓꽃 품종 분류하기

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import pandas as pd

#데이터 로드/변환
iris = load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

target_names = {
    0:iris.target_names[0],
    1:iris.target_names[1],
    2:iris.target_names[2],
}
df['target']=df['target'].map(target_names)

# print(df.head)

# 필요 데이터 추출
iris_data = df[[
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]]
iris_label = df["target"]

#학습 전용 데이터와 테스트 전용 데이터 구분
train_data,test_data,train_label,test_label=train_test_split(iris_data,iris_label)

#데이터 학습 및 예측
clf = svm.SVC()
clf.fit(train_data,train_label)
pre = clf.predict(test_data)

#정답률 출력
as_score = metrics.accuracy_score(test_label,pre)
print(f"정답률 = {as_score}")

