# 3D 렌더링 + CNN+LSTM 입문 예제 (Beginner)
# 이 노트북은 3D 메쉬(큐브, cube.off) 데이터를 GitHub에서 가져와서,
# 간단한 3D 렌더링(투영) 파이프라인을 만들고,
# 그 렌더링된 이미지를 이용해 CNN + LSTM 모델을 학습하는 입문 예제입니다.

# 대상: 비전공 학부생, 인공지능 융합 로봇 개발 입문자
# 사용 언어/프레임워크: Python, PyTorch
# 데이터:
# GitHub의 cube.off 메쉬 파일을 다운로드해서 사용
# 큐브를 회전시키며 얻은 **2D 투영 이미지 시퀀스(영상 프레임)**를 직접 생성
# 학습 문제:
# 큐브가 시계 방향으로 도는지, 반시계 방향으로 도는지(회전 방향)를 분류
# 로봇 관점에서 보면,
# 로봇의 카메라가 3D 물체(큐브)를 여러 각도에서 찍은 프레임 시퀀스를 보고
# 물체의 움직임(회전 방향)을 이해하는 아주 단순화된 예제라고 볼 수 있습니다.

# ==== 기본 라이브러리 불러오기 ====
import os  # 운영체제 기능(폴더 생성 등)을 사용하기 위한 표준 라이브러리
import math  # 수학 함수(삼각함수, 파이 등)를 사용하기 위한 표준 라이브러리
import urllib.request  # GitHub에서 파일을 다운로드하기 위한 표준 라이브러리
import random  # 난수(무작위 수)를 만들기 위한 표준 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리 (배열, 행렬 연산 등)

# PyTorch 관련 라이브러리 불러오기
import torch  # 딥러닝 연산을 위한 PyTorch 메인 패키지
import torch.nn as nn  # 신경망 레이어(CNN, LSTM 등)를 만들기 위한 모듈
import torch.optim as optim  # 옵티마이저(SGD, Adam 등)를 사용하기 위한 모듈
from torch.utils.data import Dataset, DataLoader  # 사용자 정의 데이터셋과 배치 로더를 위한 모듈

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt  # 그래프와 이미지를 그리기 위한 라이브러리

# GPU 사용 가능 여부 확인 (가능하면 GPU 사용, 아니면 CPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 현재 환경에서 사용 가능한 장치를 선택
print("사용 중인 디바이스:", device)  # 선택된 디바이스를 화면에 출력

# ==== 1단계: GitHub에서 3D 큐브 메쉬(cube.off) 파일 다운로드 ====

# GitHub raw URL (OFF 포맷의 큐브 메쉬 파일)
cube_off_url = "https://raw.githubusercontent.com/michidk/Interactive-ARAP/master/data/cube.off"  # GitHub에 있는 cube.off의 원본(raw) 주소

# 현재 작업 폴더에 데이터 저장할 디렉터리 이름 지정
data_dir = "cube_data"  # 메쉬 파일과 중간 산출물을 저장할 폴더 이름

# 폴더가 없으면 새로 생성
os.makedirs(data_dir, exist_ok=True)  # data_dir 폴더가 없으면 만들고, 이미 있으면 에러 없이 넘어감

# cube.off 파일을 저장할 경로 지정
cube_off_path = os.path.join(data_dir, "cube.off")  # data_dir 안에 cube.off라는 이름으로 저장

# GitHub에서 cube.off 파일 다운로드
urllib.request.urlretrieve(cube_off_url, cube_off_path)  # 지정한 URL에서 cube.off를 내려받아 cube_off_path에 저장

print("cube.off 다운로드 완료:", cube_off_path)  # 다운로드가 완료되었음을 출력

# 3D 좌표를 단순한 정사영(orthographic projection)으로 2D 이미지로 투영하는 함수
def render_cube_points(vertices, angle_deg, img_size=64):
    angle_rad = math.radians(angle_deg)

    # 1. X축 기준 사전 회전 (Perspective를 주기 위해 45도 기울임)
    # 큐브를 약간 기울여서 y축 높이 차이가 z축에 반영되게 함
    tilt_rad = math.radians(45.0)  # X축 기준으로 45도 고정 회전
    cos_t = math.cos(tilt_rad)
    sin_t = math.sin(tilt_rad)
    R_tilt = np.array([ # X축 회전 행렬
        [ 1.0, 0.0, 0.0 ],
        [ 0.0, cos_t, -sin_t ],
        [ 0.0, sin_t, cos_t ]
    ], dtype=np.float32)

    # 2. Y축 기준 주 회전 (요청된 각도 회전)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    R_yaw = np.array([  # Y축 회전 행렬
        [ cos_a, 0.0, sin_a],
        [ 0.0,   1.0, 0.0  ],
        [-sin_a, 0.0, cos_a]
    ], dtype=np.float32)

    # 두 회전 행렬을 곱하여 최종 회전 행렬 생성 (먼저 R_tilt, 그 다음 R_yaw)
    # >> 회전된 3D 좌표 계산 (3*3 회전 행렬)
    R = R_yaw @ R_tilt

    # 정점들을 회전 행렬 R로 변환
    # vertices : cube 의 3차원 모양을 정의하는 모든 꼭지점들의 위치 데이터
    # cube 행 개수(8) 큐브의 정점 개수
    # 열 개수 각 정점의 (x,y,z)
    # 8*3 행렬(정점 좌표 행렬)
    rotated = vertices @ R.T

    # 3. 3D -> 2D 정사영: x, z 좌표만 사용(y축 무시)
    x = rotated[:, 0]
    z = rotated[:, 2]

    # 4. 수동 정규화/스케일링 적용 (이전 수정 코드 유지)
    # 큐브는 회전 후에도 대략 [-1.5, 1.5] 범위 내에 있습니다.
    # 안전하게 [-2, 2] 범위로 간주하고 정규화를 수행하면 이미지에 꽉 차게 나옵니다.
    # 이전 [-1, 1] 범위 정규화도 작동해야 하지만, 안전을 위해 범위를 확장해 보겠습니다.

    # 큐브의 최대 범위(대각선 길이)는 약 sqrt(3) ~= 1.732
    # 안전한 정규화 범위: [-2, 2] -> [0, 1]
    # (coord + 2.0) / 4.0
    x_norm = (x + 2.0) / 4.0
    z_norm = (z + 2.0) / 4.0

    # 5. 픽셀 인덱스로 변환
    # img_size - 1 : 이미지 pixel 인덱스가 0 부터 시작
    # img_size64 >> [0,... 63]
    px = (x_norm * (img_size - 1)).astype(int)
    py = (z_norm * (img_size - 1)).astype(int)

    # 픽셀 인덱스 범위 클리핑
    px = np.clip(px, 0, img_size - 1)
    py = np.clip(py, 0, img_size - 1)

    # 6. 이미지 생성 및 블러 (기존 코드 유지)
    img = np.zeros((img_size, img_size), dtype=np.float32)
    img[py, px] = 1.0

    kernel = np.array([[0.0, 0.2, 0.0],
                       [0.2, 1.0, 0.2],
                       [0.0, 0.2, 0.0]], dtype=np.float32)

    padded = np.pad(img, 1, mode="constant")
    blurred = np.zeros_like(img)
    for i in range(img_size):
        for j in range(img_size):
            region = padded[i:i+3, j:j+3]
            blurred[i, j] = np.sum(region * kernel)

    blurred = np.clip(blurred, 0.0, 1.0)
    return blurred