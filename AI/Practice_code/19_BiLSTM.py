import torch, transformers, datasets, sklearn, evaluate, sys, platform
import re, math, random, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import Counter

SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 전처리

raw = load_dataset("imdb") # 데이터 로드: IMDb (Hugging Face datasets) # load_dataset은 자동으로 캐시 관리
token_pattern = re.compile(r"[a-z0-9']+") # [간단 토크나이저 정의(정규표현식): 영문 단어 분할
def simple_tokenize(text: str): # (한국어 주석) 소문자 변환 후 정규식 매칭
    return token_pattern.findall(text.lower())

MAX_VOCAB = 30000 # 어휘사전(Vocab) 구축 : 상위 N개 토큰 활용(희소단어 unk처리)
PAD, UNK = "<pad>", "<unk>" # 스페셜 토큰: PAD 짧은 문장이 있다면 길이를 맞추기 위해 padding token사용 # UNK 내가 사용하는 사전에 없는 단어 설정
counter = Counter()
for ex in raw["train"]: # 불용어 제거
    counter.update(simple_tokenize(ex["text"]))

most_common = counter.most_common(MAX_VOCAB - 2) # 특수 토큰 포함하여 vocab 생성 # 0번 PAD,1번 UNK 토큰 자리 남기기
itos = [PAD, UNK] + [t for t, _ in most_common]  # index-to-string # t:가장 많이 나온 단어(토큰)를 리스트로 제작
# ex: itos_ex = ['<pad>', '<unk>','i','love','you']
stoi = {t:i for i, t in enumerate(itos)} # string-to-index # 토큰:인덱스 딕셔너리 형태로 저장
PAD_IDX, UNK_IDX = stoi[PAD], stoi[UNK] #0,1

# 텍스트 -> 인덱스 변환 & 패딩/트렁케이션
MAX_LEN = 256 # (한국어 주석) 시퀀스 최대 길이 (고정)
def encode(text: str):
    tokens = simple_tokenize(text) # 영어를 소문자화
    ids = [stoi.get(tok, UNK_IDX) for tok in tokens][:MAX_LEN] # 최대 길이까지 토큰화하여 인덱스화
    # 단어가 있으면 인덱스를 가져오고 없으면 UNK_IDX(1) 반환
    if len(ids) < MAX_LEN:
        ids += [PAD_IDX] * (MAX_LEN - len(ids)) # 0=[PAD_IDX] 으로 UNKOWN부분을 ids에 채우기
    return ids

def encode_label(y: int): # (한국어 주석) IMDb: 0=neg, 1=pos # 1:긍정/0:부정으로 라벨링
    return int(y)

class IMDBTensor(torch.utils.data.Dataset): # 사용자 Dataset 정의 # torch.utils.data.Dataset 상속
    def __init__(self, hf_split):
        self.data = hf_split
    def __len__(self): # 문장 길이(데이터 개수) 반환
        return len(self.data)
    def __getitem__(self, idx): # 특정 인덱스 데이터 반환
        text = self.data[idx]["text"] # 영화 리뷰 텍스트
        label = self.data[idx]["label"] # 영화 리뷰 번호(인덱스) 0 또는 1(부정,긍정)
        x = torch.tensor(encode(text), dtype=torch.long) # 인코딩(텐서로) : 텍스트->숫자(정수)
        y = torch.tensor(encode_label(label), dtype=torch.long) # 인코딩(텐서로) : 숫자->숫자(정수)
        return x, y
train_ds = IMDBTensor(raw["train"])
test_ds  = IMDBTensor(raw["test"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2, pin_memory=torch.cuda.is_available())
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

# 모델 정의 : 임베딩 + 양방향 LSTM + FC
class BiLSTM(nn.Module): #vocab_size: 단어 길이(개수)  emb=128 : 차원 hidden=128 : 은닉층 num_layers=1 : 층 개수 num_classes=2 : 출력클래스 pad_idx=0: 패딩  
    def __init__(self, vocab_size, emb=128, hidden=128, num_layers=1, num_classes=2, pad_idx=0, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_idx) # 128차원으로 출력 [[128개],[128개],[128개],...] 각 원소는 실수값
        # I love you (vocab_size=3)-> 입력 [45,250,120] -> [[0.3,0.2,1.2,...],[0.1,0.5,1.3,...],[0.6,0.2,1.8,...]] # 128개가 안되면 패딩하여 길이 맞춤
        self.lstm = nn.LSTM(emb, hidden, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.0)
        # batch_first:배치크기를 첫번째 차원으로
        # bidirectional=양방향 순방향(문장순서대로 토큰화,앞부터 학습) 역방향(문장반대순서대로 토큰화, 뒤부터 학습)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden*2, num_classes)  # 양방향 → 2배
        # (한국어 주석) Kaiming 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight) # HE초기화 (w는 정규화)
                if m.bias is not None: nn.init.zeros_(m.bias) # HE초기화 (bias는 0으로)

    def forward(self, x):
        # x: (B, T) # 배치,시퀀스길이(탐임스탶) (ex)[64,256]
        e = self.emb(x)                 # (B, T, E) # 배치,시퀸스길이,임배딩크기 (ex)[64,256,128]
        out, (h, c) = self.lstm(e)      # h: (num_layers*2, B, H) 
        # out:각 시점(탐임스탶)의 출력값(Yt,사용안함) 
        # (h,c): h(최종 hidden state,중요) c(cell state ,내부기억,사용안함)
        # (한국어 주석) 마지막 레이어의 forward/backward hidden state 결합
        last_f = h[-2]                  # (B, H) (64, 128) 순방향 마지막
        last_b = h[-1]                  #(B, H) (64, 128) 역방향 마지막
        h_cat = torch.cat([last_f, last_b], dim=1)  # (B, 2H) (64,128)
        h_cat = self.dropout(h_cat)
        logits = self.fc(h_cat)         # (B, C) C(클래스 수) = 2 : 0,1
        return logits

model = BiLSTM(len(itos), emb=128, hidden=128, num_layers=1, pad_idx=PAD_IDX).to(device) # CPU/GPU로 이동
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 학습/평가 함수 정의
def train_one_epoch(epoch):
    model.train()
    total_loss = total_correct = total = 0 # 기본값 설정
    for X, y in train_loader:
        X, y = X.to(device), y.to(device) # 넘파이사용(CPU)  y = 정답 인덱스 번호
        optimizer.zero_grad()
        logits = model(X) #순전파
        loss = criterion(logits, y) # 손실함수
        loss.backward() # 역전파
        optimizer.step() # 파라미터 업데이트
        total_loss += loss.item() * y.size(0) # loss.item(): 평균손실. y.size(0) : 정답레이블의 배치크기
        total_correct += (logits.argmax(1) == y).sum().item() # 예측 = 정답인 경우의 수
        total += y.size(0) # 전채크기(배치의 합) 
    print(f"[Train] Epoch {epoch} | loss={total_loss/total:.4f} | acc={total_correct/total:.4f}")

@torch.no_grad() # = with torch.no_grad():
def evaluate():
    model.eval()
    total_loss = total_correct = total = 0
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    print(f"[Test ] loss={total_loss/total:.4f} | acc={total_correct/total:.4f}")

EPOCHS = 3  # 2~3 에폭 권장
for ep in range(1, EPOCHS+1):
    train_one_epoch(ep)
    evaluate()    