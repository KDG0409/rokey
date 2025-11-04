import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
# 손글씨 숫자 데이터
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1,as_frame=False)

# 이미지 데이터
image = mnist.data
# 정답 데이터
label = mnist.target
# 사이즈 지정
plt.figure(figsize=(10, 3))

# 20개 이미지를 표시
for i in range(20):

    # i번째 ax 변수 취득
    ax = plt.subplot(2, 10, i+1)

    # i번째 이미지 데이터를 취득한 다음 28x28로 변환
    img = image[i].reshape(28,28)

    # img를 이미지로 표시
    ax.imshow(img, cmap='gray_r')

    # 정답 데이터를 타이틀로 표시
    ax.set_title(label[i])

    # x, y 눈금 표시하지 않음
    ax.set_xticks([])
    ax.set_yticks([])

# 인접 객체와 겹치지 않도록 함
plt.tight_layout()

# 출력
plt.show()