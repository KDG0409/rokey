import pandas as pd # 데이터프레임 사용을 위해
from math import log # IDF 계산을 위해

# 참고자료 링크
# https://huggingface.co/datasets
# https://wikidocs.net/21695
# https://wikidocs.net/31698
# http://w.elnn.kr/search/

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
]
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

N = len(docs) # 총 문서의 수
print(N)

def tf(t, d): # 등장하는 정도
  return d.count(t)

def idf(t): # 희귀한 정도
  df = 0
  for doc in docs:
    df += t in doc
  return log(N/(df+1))

def tfidf(t, d): # 중요한 단어의 정도
  return tf(t,d)* idf(t)

result = []

# 각 문서에 대해서 아래 연산을 반복
for i in range(N):
  result.append([])
  d = docs[i]
  for j in range(len(vocab)):
    t = vocab[j]
    result[-1].append(tf(t, d))

print(result)
# [[0, 0, 0, 1, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 1]]

tf_ = pd.DataFrame(result, columns = vocab) # tf(term-frequency)
print(tf_)

#idf(inverse document frequency)
result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])
idf_

result = []
for i in range(N):
  result.append([])
  d = docs[i]
  for j in range(len(vocab)):
    t = vocab[j]
    result[-1].append(tfidf(t,d))

tfidf_ = pd.DataFrame(result, columns = vocab)
tfidf_

import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B): # 코사인 유사도(두 벡터의 유사도)
  return dot(A, B)/(norm(A)*norm(B))

doc1 = np.array([0,1,1,1])
doc2 = np.array([1,0,1,1])
doc3 = np.array([2,0,2,2])

print('문서 1과 문서2의 유사도 :',cos_sim(doc1, doc2))
print('문서 1과 문서3의 유사도 :',cos_sim(doc1, doc3))
print('문서 2와 문서3의 유사도 :',cos_sim(doc2, doc3))

