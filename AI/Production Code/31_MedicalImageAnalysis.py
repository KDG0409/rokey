# ## Kaggle API Token 업로드 코드

# 섹션 1: 환경 설정 및 라이브러리 설치
# 기본 라이브러리 임포트
import os  # 파일 및 디렉토리 관리
import numpy as np  # 수치 계산
import matplotlib.pyplot as plt  # 시각화
from PIL import Image  # 이미지 처리
import torch  # PyTorch 메인 라이브러리
import torch.nn as nn  # 신경망 모듈
import torch.optim as optim  # 최적화 알고리즘
from torch.utils.data import Dataset, DataLoader  # 데이터 로딩
import torchvision.transforms as transforms  # 이미지 변환
from torchvision import models  # 사전학습된 모델
from torchvision.models import ResNet18_Weights # 최신 가중치 로드 방식
from sklearn.metrics import confusion_matrix, classification_report  # 평가 지표
import seaborn as sns  # 시각화 (confusion matrix용)
import pydicom  # DICOM 파일 읽기
# Grad-CAM 관련 라이브러리 임포트 (모듈명은 pytorch_grad_cam을 유지)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 랜덤 시드 고정 (재현 가능한 결과를 위해)
SEED = 42
torch.manual_seed(SEED)  # PyTorch 시드
np.random.seed(SEED)  # NumPy 시드
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True # 결정적 알고리즘 사용
    torch.backends.cudnn.benchmark = False   # 속도보다는 재현성 우선

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")
print(f"PyTorch 버전: {torch.__version__}")

# 섹션 2: DICOM 기초 이해

# DICOM이란?
# Digital Imaging and Communications in Medicine
# 의료영상의 표준 포맷으로, 이미지 데이터와 메타데이터(환자정보, 촬영정보 등)를 포함

# 간단한 DICOM 구조 시각화
print("\nDICOM 파일 구조:")
print("┌─────────────────────┐")
print("│  DICOM File Header  │  ← 파일 식별자")
print("├─────────────────────┤")
print("│   Meta Information  │  ← 전송 구문 정보")
print("├─────────────────────┤")
print("│  Patient Info       │  ← 환자 정보")
print("│  Study Info         │  ← 검사 정보")
print("│  Series Info        │  ← 시리즈 정보")
print("│  Image Info         │  ← 영상 정보")
print("├─────────────────────┤")
print("│   Pixel Data        │  ← 실제 이미지 데이터")
print("└─────────────────────┘")

# 실제 DICOM 예제 (샘플 파일이 있다면)
print("\nDICOM 메타데이터 주요 항목:")
print("- PatientName: 환자 이름")
print("- PatientAge: 환자 나이")
print("- Modality: 촬영 방식 (X-Ray, CT, MRI 등)")
print("- WindowCenter/WindowWidth: 영상 밝기 조절 값")
print("- PixelData: 실제 이미지 픽셀 배열")

# 섹션 3: 데이터셋 다운로드 및 탐색

# Kaggle API 설정 (Kaggle 계정 필요)
# Kaggle에서 API Token을 다운로드하여 업로드해야 함


# Kaggle API 설정 및 데이터셋 다운로드
try:
    # files.upload() 대신 파일이 업로드되었다고 가정하고 셸 명령 실행
    # (Colab 환경에서 블록킹 오류를 방지하기 위함)
    if os.path.exists("kaggle.json"):

        print("다운로드 완료!")
        data_dir = "./data/chest_xray"
    else:
        print("'kaggle.json' 파일을 찾을 수 없습니다. 파일을 업로드해주세요.")
        data_dir = "./data/chest_xray" # 경로 변수는 설정

except Exception as e:
    print(f"다운로드 실패: {e}")
    print("   수동으로 데이터셋을 다운로드하여 업로드해주세요.")
    data_dir = "./data/chest_xray" # 실패하더라도 경로 변수는 설정

# 데이터셋 구조 확인
print("\n데이터셋 구조:")
if os.path.exists(data_dir):
    # TRAIN 디렉토리만 확인하여 경로 오류 방지
    check_dir = os.path.join(data_dir, "train")
    if os.path.exists(check_dir):
        # TRAIN, TEST 디렉토리를 중심으로 구조 출력
        for split in ['train', 'test']:
            print(f"  {split.upper()}/")
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_path = os.path.join(data_dir, split, class_name)
                if os.path.exists(class_path):
                    files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
                    print(f"    {class_name}/")
                    for file in files[:3]:  # 처음 3개만 표시
                         print(f"      {file}")
                    if len(files) > 3:
                        print(f"      ... (총 {len(files)}개 파일)")

# 데이터셋 통계
if os.path.exists(os.path.join(data_dir, "train/NORMAL")):
    train_normal = len(os.listdir(os.path.join(data_dir, "train/NORMAL")))
    train_pneumonia = len(os.listdir(os.path.join(data_dir, "train/PNEUMONIA")))
    test_normal = len(os.listdir(os.path.join(data_dir, "test/NORMAL")))
    test_pneumonia = len(os.listdir(os.path.join(data_dir, "test/PNEUMONIA")))

    print("\n데이터셋 통계:")
    print(f"   Train - NORMAL: {train_normal}장")
    print(f"   Train - PNEUMONIA: {train_pneumonia}장")
    print(f"   Test - NORMAL: {test_normal}장")
    print(f"   Test - PNEUMONIA: {test_pneumonia}장")

    # 샘플 이미지 시각화를 위한 경로 유효성 검사
    if train_normal > 0 and train_pneumonia > 0:
        # 샘플 이미지 시각화
        print("\n샘플 이미지 확인:")
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # Normal 샘플 3개
        normal_path = os.path.join(data_dir, "train/NORMAL")
        normal_images = [f for f in os.listdir(normal_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))][:3]
        for i, img_name in enumerate(normal_images):
            img = Image.open(os.path.join(normal_path, img_name))
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f"NORMAL - {img_name}", fontsize=10)
            axes[0, i].axis('off')

        # Pneumonia 샘플 3개
        pneumonia_path = os.path.join(data_dir, "train/PNEUMONIA")
        pneumonia_images = [f for f in os.listdir(pneumonia_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))][:3]
        for i, img_name in enumerate(pneumonia_images):
            img = Image.open(os.path.join(pneumonia_path, img_name))
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(f"PNEUMONIA - {img_name}", fontsize=10)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig('sample_xray_images.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("샘플 이미지가 저장되었습니다: sample_xray_images.png")
    else:
         print("\n샘플 이미지를 시각화할 데이터가 부족합니다.")

# 섹션 4: 데이터 전처리 및 Dataset 클래스

# 이미지 전처리 파이프라인 정의
# ResNet18은 ImageNet으로 사전학습되었으므로 동일한 정규화 사용
print("\n이미지 전처리 설정:")
print("   - 크기 조정: 224x224 (ResNet 입력 크기)")
print("   - RGB 변환: 흑백 → 컬러 (3채널)")
print("   - 텐서 변환 및 정규화")

# ImageNet의 평균 및 표준편차 (재사용을 위해 정의)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Train 용 전처리 (데이터 증강 포함)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.Grayscale(num_output_channels=3),  # 흑백 이미지를 컬러(RGB)로 변환
    transforms.RandomHorizontalFlip(p=0.3),  # (30% 확률로)랜덤 수평 뒤집기(좌우 반전) 데이터 증강
    transforms.RandomRotation(degrees=10),   # +- 10 도 범위 내에서 회전

    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) #ImageNet 평균값, 표준편차로 정규화
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.Grayscale(num_output_channels=3),  # 흑백 이미지를 컬러(RGB)로 변환

    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) #ImageNet 평균값, 표준편차로 정규화
])

# Custom Dataset 클래스 정의
class ChestXrayDataset(Dataset):
    """
    흉부 X-Ray 데이터셋 클래스
    NORMAL과 PNEUMONIA 두 클래스를 분류
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 데이터 루트 디렉토리 경로
            transform: 이미지 전처리 파이프라인
        """
        self.root_dir = root_dir  # 데이터 경로
        self.transform = transform  # 전처리 함수
        self.images = []  # 이미지 경로 리스트
        self.labels = []  # 라벨 리스트

        # 클래스별 디렉토리에서 이미지 파일 경로 수집
        # 0: NORMAL, 1: PNEUMONIA
        classes = ['NORMAL', 'PNEUMONIA']
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)  # 클래스 디렉토리
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):  # 이미지 파일만
                        self.images.append(os.path.join(class_dir, img_name))  # 경로 저장
                        self.labels.append(label)  # 라벨 저장
            else:
                 pass # 경로가 없으면 무시하고 넘어감

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 이미지와 라벨 반환
        Args:
            idx: 데이터 인덱스
        Returns:
            image: 전처리된 이미지 텐서
            label: 클래스 라벨 (0 or 1)
        """
        img_path = self.images[idx]  # 이미지 경로
        image = Image.open(img_path).convert('RGB')  # 이미지 로드 및 RGB 변환
        label = self.labels[idx]  # 라벨

        if self.transform:
            image = self.transform(image)  # 전처리 적용

        return image, label
    
# Dataset 및 DataLoader 생성
train_dataset = ChestXrayDataset(
    root_dir = os.path.join(data_dir, "train"),
    transform=train_transform
)

test_dataset = ChestXrayDataset(
    root_dir = os.path.join(data_dir, "test"),
    transform=test_transform
)

# 데이터가 없는 경우를 대비하여 조건문 추가
if len(train_dataset) == 0:
    print("학습 데이터셋이 비어있습니다. 이후 학습 및 평가 단계를 건너뛸 수 있습니다.")

# DataLoader 생성 (배치 단위로 데이터 로딩)
batch_size = 32  # 한 번에 처리할 이미지 개수
num_workers = 2  # 병렬 처리 워커 수 (Colab 환경에 적합)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)
print(f"   Train 데이터: {len(train_dataset)}장")
print(f"   Test 데이터: {len(test_dataset)}장")
print(f"   Batch size: {batch_size}")
print(f"   Train batches: {len(train_loader)}개")
print(f"   Test batches: {len(test_loader)}개")

# 섹션 5: 모델 구성 (ResNet18 Transfer Learning)

print("\nTransfer Learning이란?")
print("   - 대규모 데이터셋(ImageNet)으로 사전학습된 모델을 활용")
print("   - 마지막 분류층(FC)만 우리 문제에 맞게 교체")

# (사전 학습된) 모델 로드
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# 모델 구조 확인
print("\nResNet18 기본 구조:")
print(f"   - 입력: 224x224x3 RGB 이미지")
print(f"   - 출력: 1000개 클래스 (ImageNet)")

# 마지막 fc layer 교체 (1000 >> 2개 (정상, 폐렴 환자))
num_features = model.fc.in_features # fc layer 입력 특징 개수
model.fc = nn.Linear(num_features, 2)  # 2개 클래스 변경

model = model.to(device)

# 모델 학습

# 손실함수
criterion = nn.CrossEntropyLoss() # 다중 클래스 분류용 손실함수
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("\n학습 설정:")
print(f"   - 손실 함수: CrossEntropyLoss")
print(f"   - 옵티마이저: Adam")
print(f"   - 학습률: 0.0001")
print(f"   - 에폭 수: 5")

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    """
    모델 학습 함수
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        num_epochs: 학습 에폭 수
    Returns:
        train_losses: 에폭별 학습 손실
        train_accuracies: 에폭별 학습 정확도
    """
    train_losses = []  # 에폭별 손실 저장
    train_accuracies = []  # 에폭별 정확도 저장

    for epoch in range(num_epochs):
        model.train()  # 학습 모드
        running_loss = 0.0  # 현재 에폭 손실 누적
        correct = 0  # 정답 개수
        total = 0  # 전체 샘플 개수

        # 배치별 학습
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)  # 이미지를 GPU로
            labels = labels.to(device)  # 라벨을 GPU로

            # Forward pass (순전파)
            outputs = model(images)  # 모델 예측
            loss = criterion(outputs, labels)  # 손실 계산

            # Backward pass (역전파)
            optimizer.zero_grad()  # 기울기 초기화
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트

            # 통계 계산
            running_loss += loss.item()  # 손실 누적
            _, predicted = torch.max(outputs.data, 1)  # 예측 클래스
            total += labels.size(0)  # 전체 샘플 수
            correct += (predicted == labels).sum().item()  # 정답 개수

            # 100 배치마다 진행상황 출력 (데이터셋 크기 고려)
            if (batch_idx + 1) % 100 == 0:
                print(f"   Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        # 에폭 통계
        epoch_loss = running_loss / len(train_loader)  # 평균 손실
        epoch_acc = 100 * correct / total  # 정확도
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"\nEpoch [{epoch+1}/{num_epochs}] 완료:")
        print(f"   평균 Loss: {epoch_loss:.4f}")
        print(f"   정확도: {epoch_acc:.2f}%\n")

    return train_losses, train_accuracies

# 모델 학습 실행 (데이터가 있을 경우에만)
if len(train_dataset) > 0:
    print("\n학습 시작!\n")
    num_epochs = 5  # 초급 실습이므로 5 에폭
    train_losses, train_accuracies = train_model(
        model, train_loader, criterion, optimizer, num_epochs
    )
    print("\n학습 완료!")

    # 학습 곡선 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 그래프
    axes[0].plot(range(1, num_epochs+1), train_losses, 'b-o', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Accuracy 그래프
    axes[1].plot(range(1, num_epochs+1), train_accuracies, 'g-o', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("학습 곡선이 저장되었습니다: training_curves.png")
else:
    print("\n학습 데이터셋이 비어있어 학습을 건너뜁니다.")
    train_losses = []
    train_accuracies = []
