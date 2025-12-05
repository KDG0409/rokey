# 데이터셋: GitHub에서 제공하는 Flickr8k 데이터셋 (일부 샘플만 사용)
# 인코더(Encoder): 사전 학습된 ResNet-18(CNN)
# 디코더(Decoder): LSTM 기반 문장 생성기

# ===== 1. 기본 라이브러리 임포트 =====
import os  # 운영체제 기능(폴더 생성, 경로 처리 등)을 사용하기 위한 모듈
import re  # 정규표현식(텍스트 전처리)에 사용되는 모듈
import zipfile  # zip 파일(압축 파일)을 풀기 위해 사용하는 모듈
import random  # 무작위 샘플 추출, 시드 고정 등에 사용하는 모듈
from collections import Counter  # 단어 빈도수를 세기 위해 사용하는 자료구조

import urllib.request  # 인터넷에서 파일을 다운로드하기 위한 표준 라이브러리 모듈

import numpy as np  # 숫자 계산과 배열 연산을 편리하게 해주는 라이브러리
from PIL import Image  # 이미지 파일을 열고 다루기 위한 라이브러리(Pillow)
import matplotlib.pyplot as plt  # 그래프나 이미지를 화면에 출력하기 위한 라이브러리

import torch  # PyTorch 딥러닝 프레임워크의 핵심 패키지
from torch import nn  # 신경망 레이어를 만들기 위한 모듈
from torch.utils.data import Dataset, DataLoader  # 데이터셋과 배치 생성을 도와주는 클래스들

from torchvision import transforms  # 이미지 전처리(리사이즈, 텐서 변환 등)를 위한 모듈
from torchvision.models import resnet18, ResNet18_Weights  # 사전 학습된 ResNet-18 모델과 그 가중치 설정

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU가 있으면 GPU, 없으면 CPU를 사용하도록 설정

print("사용 중인 디바이스:", device)  # 현재 사용 중인 디바이스를 출력하여 확인

import os
import urllib.request # urllib.request가 정의되지 않았으므로 추가
import zipfile # zipfile이 정의되지 않았으므로 추가

# ===== 3. GitHub에서 Flickr8k 데이터 다운로드 및 압축 해제 =====
data_dir = "./flickr8k"  # 전체 데이터셋을 저장할 기본 폴더 경로를 문자열로 지정
os.makedirs(data_dir, exist_ok=True)  # 위에서 지정한 폴더가 없으면 새로 만들고, 있으면 그대로 둠

# GitHub 릴리즈에서 제공되는 이미지(zip)와 캡션 텍스트(zip)의 URL
images_zip_url = "https://github.com/Avaneesh40585/Flickr8k-Dataset/releases/download/v1.0/Flickr8k_Dataset.zip"  # 이미지 압축 파일의 인터넷 주소
text_zip_url = "https://github.com/Avaneesh40585/Flickr8k-Dataset/releases/download/v1.0/Flickr8k_text.zip"  # 캡션 텍스트 압축 파일의 인터넷 주소

images_zip_path = os.path.join(data_dir, "Flickr8k_Dataset.zip")  # 이미지 zip 파일을 로컬에 저장할 경로
text_zip_path = os.path.join(data_dir, "Flickr8k_text.zip")  # 텍스트 zip 파일을 로컬에 저장할 경로

def download_if_not_exists(url, save_path):  # 파일이 없을 때만 다운로드하는 함수 정의
    if not os.path.exists(save_path):  # 지정한 경로에 파일이 존재하지 않는다면
        print(f"다운로드 중: {url}")  # 어떤 URL을 다운로드 중인지 출력
        urllib.request.urlretrieve(url, save_path)  # urlretrieve 함수를 사용하여 URL에서 파일을 받아 로컬에 저장
        print("완료:", save_path)  # 다운로드가 끝나면 완료 메시지 출력
    else:
        print("이미 존재함:", save_path)  # 이미 파일이 있다면 다운로드를 생략하고 메시지만 출력

download_if_not_exists(images_zip_url, images_zip_path)  # Flickr8k 이미지 zip 파일을 다운로드(필요할 때만)
download_if_not_exists(text_zip_url, text_zip_path)  # Flickr8k 텍스트 zip 파일을 다운로드(필요할 때만)

# zip 파일을 실제 폴더로 압축 해제하는 함수 정의
def unzip_if_needed(zip_path, extract_to):  # zip 파일 경로와 압축을 풀 폴더 경로를 인자로 받음
    # 압축 해제된 폴더의 예상 이름을 미리 확인
    expected_folder_name = os.path.splitext(os.path.basename(zip_path))[0]
    expected_extract_path = os.path.join(extract_to, expected_folder_name)

    # 압축 해제된 폴더가 아직 없다면
    if not os.path.exists(expected_extract_path):
        print(f"압축 해제 중: {zip_path}")  # 어느 zip 파일을 풀고 있는지 출력
        with zipfile.ZipFile(zip_path, "r") as zf:  # zipfile.ZipFile을 사용해 zip 파일을 읽기 모드로 엶
            zf.extractall(extract_to)  # 지정한 폴더로 모든 파일을 압축 해제
        print("압축 해제 완료:", expected_extract_path)  # 완료 메시지 출력
    else:
        print("이미 압축 해제됨:", expected_extract_path)  # 이미 폴더가 있다면 다시 풀지 않고 메시지만 출력

# 압축 해제는 data_dir (./flickr8k)에 이루어지므로,
# 실제 데이터셋 폴더는 ./flickr8k/Flickr8k_Dataset 에 위치하게 됩니다.
unzip_if_needed(images_zip_path, data_dir)  # 이미지 zip 파일을 flickr8k 폴더 아래에 압축 해제
unzip_if_needed(text_zip_path, data_dir)  # 텍스트 zip 파일을 flickr8k 폴더 아래에 압축 해제

images_folder = os.path.join("flickr8k", "Flickr8k_Dataset", "Flickr8k_Dataset")
print("이미지 폴더 존재 여부:", os.path.exists(images_folder))

# 캡션 파일 로드 및 구조 이해
import zipfile

# 데이터셋 압축 파일 경로 (이미지에서 보인 경로를 기반으로 예시)
# 'data_dir'이 'flickr8k'의 상위 폴더를 가리킨다고 가정합니다.
zip_path = os.path.join(data_dir, "Flickr8k_text.zip")

# 압축을 풀 디렉토리
extract_to_dir = data_dir

# 압축 해제
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_dir)
    print(f"'{zip_path}' 파일이 '{extract_to_dir}'에 성공적으로 압축 해제되었습니다.")

# 텍스트 전처리 및 이미지-캡션 매핑 만들기
# 한 줄씩 읽어 이미지 이름과 문장 부분을 분리합니다.
# 문장 안의 불필요한 기호(쉼표, 마침표 등)를 제거하고, 모두 소문자로 바꿉니다.
# 이미지 이름을 key로 하고, 그 이미지에 대한 여러 캡션 리스트를 value로 갖는 딕셔너리를 만듭니다.
def clean_sentence(sentence: str) -> str:  # 한 문장을 깨끗하게 전처리하는 함수 정의
    sentence = sentence.lower()  # 모든 문자를 소문자로 변환 (예: 'A Dog' -> 'a dog')
    sentence = re.sub(r"[^a-z ]", "", sentence)  # 알파벳 소문자와 공백을 제외한 문자(숫자, 기호 등)를 제거
    sentence = re.sub(r"\s+", " ", sentence).strip()  # 여러 개의 공백을 하나로 줄이고, 양끝 공백 제거
    return sentence  # 전처리가 끝난 문장을 반환

file_path = '/content/flickr8k/Flickr8k.token.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

captions_dict = {}  # 이미지 파일 이름을 key, 해당 이미지의 문장 리스트를 value로 저장할 딕셔너리

for line in lines:  # 캡션 파일에서 읽어온 모든 줄을 하나씩 순회
    line = line.strip()  # 줄 끝의 줄바꿈 문자 등을 제거하여 깔끔한 문자열로 만듦
    if len(line) == 0:  # 빈 줄인 경우는 건너뛰기
        continue  # 다음 줄로 넘어감
    image_and_caption = line.split("\t")  # 탭 문자 기준으로 이미지 정보와 문장을 분리
    if len(image_and_caption) != 2:  # 만약 탭으로 나눈 결과가 2개가 아니라면 형식이 이상한 것이므로
        continue  # 해당 줄은 건너뛰고 다음 줄로 이동
    image_id_raw, caption_raw = image_and_caption  # 왼쪽은 이미지+번호, 오른쪽은 문장 부분으로 변수에 저장
    image_filename = image_id_raw.split("#")[0]  # '파일이름#번호' 형태에서 앞부분(파일 이름)만 사용
    cleaned = clean_sentence(caption_raw)  # 위에서 정의한 함수로 문장을 전처리
    if len(cleaned.split()) < 3:  # 단어 수가 너무 적은 문장은 학습에 별 도움이 안 되므로 제외
        continue  # 다음 줄로 넘어감
    captions_dict.setdefault(image_filename, []).append(cleaned)  # 해당 이미지 파일 이름에 문장 추가

print("이미지 개수(캡션 포함):", len(captions_dict))  # 캡션이 있는 이미지가 몇 개인지 출력

# 한 이미지에 어떤 캡션들이 들어 있는지 예시로 하나만 출력
sample_key = next(iter(captions_dict.keys()))  # 딕셔너리에서 임의의 첫 번째 key를 가져옴
print("예시 이미지 파일 이름:", sample_key)  # 선택된 이미지 파일 이름 출력
print("해당 이미지의 캡션들:")  # 그 이미지에 대응되는 문장들을 출력하겠다는 안내 메시지
for c in captions_dict[sample_key]:  # 선택된 이미지에 대한 캡션 리스트를 순회
    print("-", c)  # 각 캡션을 한 줄에 하나씩 출력

# 작은 서브셋 사용하기

all_image_filenames = list(captions_dict.keys())

subset_size = 200
if len(all_image_filenames) < subset_size:
    subset_size = len(all_image_filenames)

small_image_filenames = random.sample(all_image_filenames, subset_size)
len(small_image_filenames)

# 단어 사전(vocabulary) 만들기
# 이미지 캡셔닝에서 문장을 다룰려면 단어 >> 숫자(index) 바꿔야 함
# 특별토큰(special token)
# pad , start, end, unk
# ===== 단어 사전 구성 =====
special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]  # 특별한 의미를 가지는 4개의 특수 토큰 리스트

word_counter = Counter()  # 각 단어가 몇 번 등장했는지 세기 위한 Counter 객체
for img in small_image_filenames:  # 선택된 서브셋 이미지들에 대해서만 반복
    for cap in captions_dict[img]:  # 각 이미지에 대해 여러 캡션들을 순회
        for w in cap.split():  # 문장을 공백 기준으로 나누어 단어 리스트를 얻음
            word_counter[w] += 1  # 해당 단어의 등장 빈도를 1 증가시킴

min_freq = 3  # 단어가 최소 몇 번 이상 나타나야 사전에 포함할지 기준 (여기서는 3번 이상)
vocab_words = [w for w, c in word_counter.items() if c >= min_freq]  # 등장 빈도가 기준 이상인 단어만 추려서 리스트 생성
print("기준 이상으로 등장한 단어 수:", len(vocab_words))  # 사전에 포함될 일반 단어 수를 출력

idx2word = []  # 인덱스에서 단어로 바꾸기 위한 리스트(인덱스 -> 단어)
idx2word.extend(special_tokens)  # 앞쪽에 특수 토큰들을 순서대로 추가
idx2word.extend(sorted(vocab_words))  # 나머지 단어들을 정렬하여 뒤에 붙임

word2idx = {w: i for i, w in enumerate(idx2word)}  # 단어에서 인덱스로 바꾸기 위한 딕셔너리(단어 -> 인덱스)

pad_idx = word2idx["<pad>"]  # 패딩 토큰의 인덱스를 변수로 저장 (나중에 손실 계산에서 무시할 때 사용)
start_idx = word2idx["<start>"]  # 문장 시작 토큰의 인덱스
end_idx = word2idx["<end>"]  # 문장 끝 토큰의 인덱스
unk_idx = word2idx["<unk>"]  # 사전에 없는 단어를 대신할 토큰의 인덱스

vocab_size = len(idx2word)  # 최종 단어 사전의 크기(특수 토큰 포함)
print("최종 단어 사전 크기:", vocab_size)  # 사전 크기를 출력

# ===== 문장을 숫자 시퀀스로 변환하는 함수 =====
def sentence_to_indices(sentence: str, max_len: int = 20):  # 문장과 최대 길이를 받아서 인덱스 리스트로 변환하는 함수
    tokens = sentence.split()  # 공백 기준으로 단어들을 분리하여 리스트로 만듦
    indices = [start_idx]  # 문장 시작을 의미하는 토큰 인덱스를 맨 앞에 추가
    for w in tokens:  # 문장의 각 단어에 대해 반복
        idx = word2idx.get(w, unk_idx)  # 단어가 사전에 있으면 그 인덱스를, 없으면 <unk> 인덱스를 가져옴
        indices.append(idx)  # 인덱스 리스트에 추가
        if len(indices) >= max_len - 1:  # 이미 충분히 길어졌다면 (마지막에 <end>를 하나 더 붙일 예정)
            break  # 더 이상 단어를 추가하지 않고 반복 종료
    indices.append(end_idx)  # 문장 끝을 의미하는 <end> 토큰 인덱스를 마지막에 추가
    # 길이가 너무 짧으면 뒤쪽을 <pad> 인덱스로 채워서 길이를 맞춤
    if len(indices) < max_len:  # 현재 길이가 최대 길이보다 짧다면
        indices.extend([pad_idx] * (max_len - len(indices)))  # 남은 부분을 모두 <pad>로 채움
    return indices  # 완성된 인덱스 리스트를 반환

# 예시로 한 문장을 숫자 시퀀스로 변환해 보기
example_sentence = captions_dict[small_image_filenames[0]][0]  # 서브셋의 첫 번째 이미지에 대한 첫 번째 캡션 문장을 가져옴
print("예시 원본 문장:", example_sentence)  # 원본 문장을 출력
example_indices = sentence_to_indices(example_sentence, max_len=10)  # 최대 길이를 10으로 제한하여 인덱스 시퀀스로 변환
print("숫자 시퀀스:", example_indices)  # 변환된 인덱스 리스트를 출력
print("다시 단어로:", [idx2word[i] for i in example_indices])  # 인덱스를 다시 단어로 바꿔서 사람이 읽을 수 있게 출력

# PyTorch Dataset 만들기 (이미지 + 캡션)
# 딥러닝 학습을 위해서는 데이터를 (입력, 정답) 형태로 계속 공급해 주어야 합니다.

# 입력(Input): 전처리된 이미지 텐서
# 정답(Target): 같은 이미지에 대한 캡션 문장(숫자 시퀀스)
# PyTorch의 Dataset 클래스를 상속하여, 우리가 원하는 형식으로 데이터를 꺼낼 수 있도록 만들어 봅니다.

# ===== PyTorch Dataset 정의 =====
class FlickrDataset(Dataset):  # PyTorch의 Dataset을 상속하여 FlickrDataset이라는 새로운 클래스 정의
    def __init__(self, image_folder, image_filenames, captions_dict, transform=None, max_len: int = 20):  # 생성자 정의
        self.image_folder = image_folder  # 이미지 파일이 저장된 폴더 경로를 멤버 변수로 저장
        self.image_filenames = image_filenames  # 사용할 이미지 파일 이름 리스트를 멤버 변수로 저장
        self.captions_dict = captions_dict  # 이미지별 캡션 정보를 담고 있는 딕셔너리를 멤버 변수로 저장
        self.transform = transform  # 이미지에 적용할 전처리(transform)를 멤버 변수로 저장
        self.max_len = max_len  # 캡션 최대 길이를 멤버 변수로 저장

        # (이미지, 캡션) 쌍을 미리 펼쳐서 리스트로 저장
        self.samples = []  # 모든 (이미지 파일 이름, 캡션 문자열)을 저장할 리스트
        for img in self.image_filenames:  # 선택된 이미지 파일 이름들을 하나씩 순회
            for cap in self.captions_dict[img]:  # 각 이미지에 대해 여러 개의 캡션을 순회
                self.samples.append((img, cap))  # (이미지 파일 이름, 캡션 문자열) 튜플을 samples 리스트에 추가

    def __len__(self):  # 데이터셋의 길이를 반환하는 메서드(필수 구현)
        return len(self.samples)  # (이미지, 캡션) 쌍의 총 개수를 반환

    def __getitem__(self, idx):  # 인덱스로 데이터 하나를 꺼내는 메서드(필수 구현)
        img_name, caption = self.samples[idx]  # samples 리스트에서 idx 위치의 (이미지 파일 이름, 캡션)을 가져옴
        img_path = os.path.join(self.image_folder, img_name)  # 이미지 파일이 실제로 있는 전체 경로를 만듦

        image = Image.open(img_path).convert("RGB")  # 이미지 파일을 열고, RGB 3채널 이미지로 변환
        if self.transform is not None:  # 만약 transform이 정의되어 있다면
            image = self.transform(image)  # 이미지에 전처리(transform)를 적용

        caption_indices = sentence_to_indices(caption, max_len=self.max_len)  # 문자열 캡션을 숫자 인덱스 시퀀스로 변환
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)  # 리스트를 PyTorch LongTensor로 변환

        return image, caption_tensor  # (이미지 텐서, 캡션 텐서)를 튜플로 반환
    
# ===== 이미지 전처리(transform)와 DataLoader 정의 =====
image_transform = transforms.Compose([  # 여러 전처리 과정을 순서대로 적용하기 위한 Compose 사용
    transforms.Resize((224, 224)),  # 이미지를 224x224 크기로 리사이즈 (ResNet 입력 크기에 맞춤)
    transforms.ToTensor(),  # 이미지를 [0,1] 범위의 PyTorch 텐서(채널, 높이, 너비)로 변환
    transforms.Normalize(  # 이미지의 픽셀 값을 평균 0, 표준편차 1 근처로 맞추기 위한 정규화
        mean=[0.485, 0.456, 0.406],  # ImageNet 데이터셋에서 계산된 채널별 평균값
        std=[0.229, 0.224, 0.225],  # ImageNet 데이터셋에서 계산된 채널별 표준편차
    ),
])

max_caption_len = 20  # 캡션의 최대 길이를 20 단어로 제한

dataset = FlickrDataset(  # 위에서 정의한 FlickrDataset 클래스를 사용해 데이터셋 인스턴스를 생성
    image_folder=images_folder,  # 이미지가 저장된 폴더 경로 전달
    image_filenames=small_image_filenames,  # 사용할 이미지 파일 이름 리스트 전달
    captions_dict=captions_dict,  # 이미지별 캡션 딕셔너리 전달
    transform=image_transform,  # 이미지 전처리(transform) 전달
    max_len=max_caption_len,  # 캡션 최대 길이 전달
)

print("(이미지, 캡션) 샘플 수:", len(dataset))  # 데이터셋에 몇 개의 (이미지, 캡션) 쌍이 있는지 출력

batch_size = 16  # 한 번에 모델에 넣을 데이터 개수(배치 크기)를 16으로 설정

dataloader = DataLoader(  # PyTorch DataLoader를 사용하여 배치 단위로 데이터를 꺼낼 수 있도록 준비
    dataset,
    batch_size=batch_size,  # 위에서 설정한 배치 크기 사용
    shuffle=True,  # 매 epoch마다 데이터 순서를 섞어서 학습이 편향되지 않도록 설정
)

# CNN 인코더 정의
# 이미지 입력 받아 >> 특징(feauture) 그 이미지의 특징을 요약한 벡터 (feature vector) 만들어줘요
# 사전학습된 resnet18 model 이용, 마지막 분류기(clf) 레이어 부분 제거 >> 512차원 특징벡터 사용

# ===== CNN 인코더 정의 (ResNet-18) =====
class EncoderCNN(nn.Module):  # PyTorch의 nn.Module을 상속하여 이미지 인코더 클래스를 정의
    def __init__(self, embed_size: int = 256):  # 임베딩 차원(embed_size)을 인자로 받아 초기화
        super().__init__()  # 부모 클래스(nn.Module)의 초기화 메서드 호출
        weights = ResNet18_Weights.DEFAULT  # torchvision에서 제공하는 ResNet-18의 기본 사전 학습 가중치 설정
        resnet = resnet18(weights=weights)  # 사전 학습된 가중치를 가진 ResNet-18 모델 불러오기
        modules = list(resnet.children())[:-1]  # 마지막 분류용 FC 레이어를 제외한 나머지 레이어들만 리스트로 추출
        self.cnn = nn.Sequential(*modules)  # 추출한 레이어들을 nn.Sequential로 묶어서 하나의 모듈로 구성
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)  # ResNet 마지막 특성 차원에서 embed_size로 줄이는 선형 레이어
        self.bn = nn.BatchNorm1d(embed_size)  # 학습 안정화를 위해 배치 정규화 레이어 추가

        for param in self.cnn.parameters():  # 사전 학습된 CNN 가중치들에 대해 반복
            param.requires_grad = False  # 입문용 예제에서는 CNN 부분은 학습하지 않고 고정(freeze)하여 빠르게 학습

    def forward(self, images):  # 순전파(forward) 메서드 정의, 입력은 이미지 텐서
        features = self.cnn(images)  # CNN을 통과시켜 (배치, 채널, 1, 1) 형태의 특징 맵을 얻음
        features = features.view(features.size(0), -1)  # (배치, 채널, 1, 1)을 (배치, 채널) 형태로 펼침
        features = self.fc(features)  # 선형 레이어를 통과시켜 embed_size 차원의 벡터로 변환
        features = self.bn(features)  # 배치 정규화로 분포를 안정화
        return features  # 최종 이미지 특징 벡터를 반환
    
# LSTM 디코더(Decoder) 정의
# 디코더는 인코더가 만든 이미지 특징 벡터와 이전까지 생성된 단어들을 이용하여, 다음 단어를 하나씩 예측하는 문장 생성기입니다.

# 단어를 임베딩(Embedding) 레이어를 통해 숫자 벡터로 바꾼 뒤,
# LSTM 에 순서대로 넣어 주고,
# LSTM의 출력을 Linear 레이어를 통해 각 단어가 나올 확률로 변환합니다.

# ===== LSTM 디코더 정의 =====
class DecoderRNN(nn.Module):  # PyTorch nn.Module을 상속하여 디코더 클래스를 정의
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):  # 초기화 메서드
        super().__init__()  # 부모 클래스 초기화
        self.embed = nn.Embedding(vocab_size, embed_size)  # 단어 인덱스를 embed_size 차원의 벡터로 바꿔주는 임베딩 레이어
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM 레이어 정의 (입력: embed_size, 은닉: hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)  # LSTM 출력을 단어 사전 크기만큼의 로짓(logit)으로 변환하는 선형 레이어

    def forward(self, features, captions):  # 순전파 메서드, features: 이미지 벡터, captions: 정답 캡션 시퀀스
        embeddings = self.embed(captions)  # (배치, 시퀀스 길이) 형태의 캡션 인덱스를 임베딩 벡터로 변환
        features = features.unsqueeze(1)  # (배치, embed_size)를 (배치, 1, embed_size)로 차원 확장하여 LSTM 첫 입력으로 사용
        inputs = torch.cat((features, embeddings[:, :-1, :]), dim=1)  # 이미지 특징 뒤에 캡션의 마지막 토큰을 제외한 부분을 이어붙여 입력 시퀀스 생성
        outputs, _ = self.lstm(inputs)  # LSTM에 입력 시퀀스를 넣어 전체 시퀀스에 대한 은닉 상태 출력
        outputs = self.fc(outputs)  # 각 시점의 LSTM 출력을 단어 사전 크기의 로짓으로 변환
        return outputs  # (배치, 시퀀스 길이, vocab_size) 형태의 예측 결과 반환

    def sample(self, features, max_len=20):  # 학습된 모델로부터 실제 문장을 생성하기 위한 메서드
        generated_indices = []  # 생성된 단어 인덱스를 순서대로 저장할 리스트
        inputs = features.unsqueeze(1)  # (배치=1, 1, embed_size) 형태로 LSTM 입력 준비
        states = None  # LSTM의 초기 은닉 상태와 셀 상태는 None으로 두면 자동 초기화
        for _ in range(max_len):  # 최대 max_len 길이만큼 단어를 생성
            outputs, states = self.lstm(inputs, states)  # 현재 입력과 상태를 LSTM에 넣어 한 시점의 출력을 얻음
            outputs = self.fc(outputs.squeeze(1))  # LSTM 출력을 선형 레이어에 통과시켜 단어별 로짓으로 변환
            _, predicted = outputs.max(1)  # 가장 확률이 높은 단어 인덱스를 선택
            generated_indices.append(predicted.item())  # 선택된 인덱스를 리스트에 추가
            if predicted.item() == end_idx:  # 만약 <end> 토큰이 나오면 문장 생성을 멈춤
                break  # 반복문 종료
            inputs = self.embed(predicted).unsqueeze(1)  # 예측된 단어를 임베딩하여 다음 시점의 입력으로 사용
        return generated_indices  # 생성된 단어 인덱스 리스트를 반환
    
# ===== 모델 인스턴스 생성 및 학습 설정 =====
embed_size = 256  # 이미지 특징 벡터와 단어 임베딩 벡터의 차원을 256으로 설정
hidden_size = 512  # LSTM 은닉 상태의 차원을 512로 설정

encoder = EncoderCNN(embed_size=embed_size).to(device)  # EncoderCNN 인스턴스를 만들고, GPU/CPU 디바이스로 이동
decoder = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size).to(device)  # DecoderRNN 인스턴스를 만들고 디바이스로 이동

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # 손실 함수로 다중 클래스 분류에 사용하는 CrossEntropyLoss를 사용, 패딩 토큰은 무시
params = list(decoder.parameters()) + [p for p in encoder.fc.parameters()] + [p for p in encoder.bn.parameters()]  # 학습할 파라미터들만 모아서 리스트로 생성
optimizer = torch.optim.Adam(params, lr=1e-3)  # Adam 옵티마이저를 사용하여 파라미터를 업데이트, 학습률은 0.001로 설정

import os
import torch
import torch.nn as nn

num_epoch = 2

encoder.train()
decoder.tratin()

for epoch in range(num_epoch):
    total_loss = 0.0 # 한 epoch 동안 손실 누적

    for images,captions in dataloader:
        images = images.to(device)
        captions = captions.to(device)
        optimizer.zero_grad()
        features = encoder(images) # 이미지 >> 인코더(cnn) >> 이미지 특징벡터
        outputs = decoder(features, captions)
        # (이미지특징, 캡션정답) >> 디코더에 넣어 단어별 예측 로짓 얻음.
        # CrossEntropyLoss(배치*seq_len, vocab_size) 손실계산
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        loss.backward() # 역전파 수행
        optimizer.step() # 가중치 업데이트
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader) # 한 epoch 동안 평균 손실 계산
    print(f"epoch [{epoch+1}/{num_epoch}], 평균손실 ")

# 간단한 학습 루프 (입문용으로 1~2 epoch만 돌려보기)

from torch.utils.data import Dataset
import os

class Flickr8kDataset(Dataset):
    def __init__(self, image_folder, captions_dict, transform=None, max_len=20):
        self.image_folder = image_folder         # 예: "./flickr8k/Flicker8k_Dataset"
        self.captions_dict = captions_dict       # {"파일명.jpg": [token_id,...], ...}
        self.transform = transform
        self.max_len = max_len

        # 1) 캡션에 등장하는 전체 이미지 파일명
        all_image_ids = list(captions_dict.keys())

        # 2) 실제 폴더에 존재하는 파일만 남기기
        valid_image_ids = []
        missing_image_ids = []

        for img_id in all_image_ids:
            img_path = os.path.join(self.image_folder, img_id)
            if os.path.exists(img_path):
                valid_image_ids.append(img_id)
            else:
                missing_image_ids.append(img_id)

        self.image_ids = valid_image_ids


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_folder, img_id)

        # 혹시 모를 예외 상황 방지용 (거의 안 나오겠지만 안전장치)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Dataset 내부 오류: {img_path} 가 존재하지 않습니다.")

        # 이미지 로드
        from PIL import Image
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # 캡션 텐서 가져오기 (이미 앞에서 토크나이즈 + 패딩했다고 가정)
        caption = self.captions_dict[img_id]

        return image, caption
    
# GitHub에서 Flickr8k 데이터셋을 다운로드하고, 이미지와 캡션을 로드했습니다.
# 텍스트를 전처리하여 단어 사전(vocabulary)을 만들고, 문장을 숫자 시퀀스로 변환했습니다.
# 사전 학습된 ResNet-18을 인코더로 사용해 이미지 특징을 추출했습니다.
# LSTM 기반 디코더를 사용해, 이미지 특징과 캡션을 이용하여 다음 단어를 예측하는 모델을 만들었습니다.
# 작은 서브셋에 대해 간단한 학습을 수행하고, 실제로 캡션을 생성해 보았습니다.