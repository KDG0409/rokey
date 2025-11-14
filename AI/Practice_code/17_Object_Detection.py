import torch, cv2, numpy as np, matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# (IoU 계산 함수) 두 박스의 교집합/합집합 비율
# opencv 버전
def compute_iou(boxA,boxB):
    # iou : 두 박스가 겹치는 정도의 비율 [0,1], 클수록 겹침
    # box = [x1,y1,x2,y2] 
    xA = max(boxA[0],boxB[0]) #겹치는 왼쪽 위 좌표
    yA = max(boxA[1],boxB[1])
    xB = min(boxA[2],boxB[2]) #겹치는 오른쪽 아래 좌표
    yB = min(boxA[3],boxB[3])
    inter = max(0,xB-xA) * max(0,yB-yA) # 겹치는 가로 길이 x 겹치는 세로 길이

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) # A면적 
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) # B면적
    union = areaA + areaB - inter
    return union / inter if union > 0 else 0

import torch
# pytorch 버전
def iou_pytorch(boxes1, boxes2):
    """
    PyTorch 버전 IoU 계산 함수
    boxes1, boxes2: (..., 4) 형태의 tensor
    box = [x1, y1, x2, y2]
    """

    # -----------------------------
    # 1. 교집합 영역 계산
    # -----------------------------

    # boxes1[..., 0], boxes2[..., 0]
    # ..., : 파이썬 _ (안 가져오겠다) 의미와 유사
    # >> 앞 쪽의 모든 차원 그대로 두고, 마지막 차원에서 인덱스 0만

    # 두 박스의 왼쪽 위 점(x1, y1) 중 큰 값 선택 → 교차 영역의 시작점
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])

    # 두 박스의 오른쪽 아래 점(x2, y2) 중 작은 값 선택 → 교차 영역의 끝점
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    # 교차 영역의 width, height (음수가 나오면 0으로 처리 → 안 겹치는 경우)
    inter_w = torch.clamp(x2 - x1, min=0)
    inter_h = torch.clamp(y2 - y1, min=0)

    # 교차 영역 면적
    inter_area = inter_w * inter_h

    # -----------------------------
    # 2. 각 박스의 면적 계산
    # -----------------------------
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # -----------------------------
    # 3. 합집합 영역
    # -----------------------------
    union = area1 + area2 - inter_area

    # 0 division 방지
    eps = 1e-7

    # -----------------------------
    # 4. IoU 계산
    # -----------------------------
    iou = inter_area / (union + eps)

    return iou

boxA = torch.tensor([10,10,60,60])
boxB = torch.tensor([30,30,80,80])

print(iou_pytorch(boxA, boxB)) # tensor(0.2195)

# 배치(batch) 단위 IoU 계산
# 이미지 32개 >> 배치 32
boxA = torch.tensor([[10,10,60,60],
                     [50,50,100,100]], dtype=torch.float)
boxB = torch.tensor([[30,30,80,80],
                     [60,60,120,120]], dtype=torch.float)

print(iou_pytorch(boxA, boxB))
print(boxA.shape) # (2,4)
# >> (N,4) : N 배치크기

print(iou_pytorch(boxA, boxB)) # tensor([0.2195, 0.3556])

# 사전학습 Faster R-CNN 모델 로드
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
det_model = fasterrcnn_resnet50_fpn(weight=weights).to(device)
det_model.eval()
preproc = weights.transforms()

# 테스트 이미지 준비(샘플 다운로드 또는 업로드)
# 간단한 단색/박스 테스트
img = Image.new('RGB', (640,480), color=(200,200,200)) # 색상(BGR)>색상(RGB)
# 사각형 그려보기
canvas = np.array(img).copy()
cv2.rectangle(canvas, (100,100), (300,300),(255,0,0),3) # 사각형 왼쪽 위 좌표 (top-left point)>사각형 오른쪽 아래 좌표 (bottom-right point)>색상(RGB)>선두께
cv2.circle(canvas,(450,450),60,(0,255,0),3) # 중심좌표,raidius,색상,두께
img = Image.fromarray(canvas) # img 업데이트
cv2.imshow(canvas)

# 전처리 -> 모델 추론
x = preproc(img).unsqueeze(0).to(device) # 맨앞 차원(배치 차원) 추가

with torch.no_grad():
    output = det_model(x)
    out = det_model(x)[0] # 리스트를 벗긴 값 >예측값: 딕셔너리형태
    boxes = out['boxes'].cpu().numpy() # 넘파이 변환/CPU변경 : boxes만 출력 /박스 좌표
    labels = out['labels'].cpu().numpy() # 예측 클래스
    scores = out['scores'].cpu().numpy() # 신뢰도(확률)

# 신뢰도 임계치로 필터링
thr = 0.5 # threshold (임계치)
keep = scores >= thr
boxes = boxes[keep]
labels = labels[keep]
scores = scores[keep]

# 시각화
vis = np.array(img).copy()
for (x1,y1,x2,y2), s, lb in zip(boxes, scores, labels):
    cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
    cv2.putText(vis, f"id:{int(lb)} {s:.2f}", (int(x1),int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2) # 크기,색상,두께
plt.figure(figsize=(8,6)); plt.imshow(vis); plt.axis('off'); plt.title('Faster R-CNN Prediction'); plt.show()