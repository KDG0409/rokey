import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 강의_6기_AI개론_1차시_01_intro_완성코드 이미지 분류 코드 
# 파일 생성 및 데이터 생성 후 정규화,증강, 배치 사이즈로 데이터 로더 적용 -> 전이학습(공통함수)

# 초기설정
# warning 표시 끄기
import warnings
warnings.simplefilter('ignore')
# 기본 폰트 설정
# plt.rcParams['font.family'] = font_name
# GPU 디바이스 할당 (8장)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 공통함수 읽어오기
