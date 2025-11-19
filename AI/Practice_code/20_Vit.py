import torch, transformers, datasets, evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification
import os
from transformers import TrainingArguments, Trainer, DefaultDataCollator
import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
beans=load_dataset("beans")

# 사전정의모델 로드
MODEL = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(MODEL)

# 라벨 매핑 설정
key = "labels" if "labels" in beans["train"].features else "label"
# 데이터 셋에 labels가 있으면 key=labels, 없으면 key=label(사전정의에 따른 오류방지)
names = beans["train"].features[key].names # features[key] = value값
id2label = {i:n for i,n in enumerate(names)}
label2id = {n:i for i,n in enumerate(names)}

# 모델 로드
model = ViTForImageClassification(
    MODEL, num_labels = len(names),
    id2label = id2label, label2id = label2id,
    ignore_mismatched_sizes = True # 1000개 분류모델로 3개 분류하니까 에러->크기 안맞는부분 무시
)

# 코랩의 CPU 코어 개수 확인 (보통 2~4개)
num_proc = os.cpu_count()

# 전처리 정의
def transform(ex):
    inputs = processor(images = ex['image'], return_tensors='pt')
    ex["pixel_values"] = inputs["pixel_values"]
    # 원본 데이터에 변환된 이미지 데이터 추가(새로 열 생성)
    return ex

beans = beans.map(transform,batched=True,num_proc=num_proc,remove_columns=['image'])
beans.set_format("torch")

def keep(split): # 필요한 컬럼(열,특성)만 유지하는 사용자 함수
    cols = ['pixel_values',key] # 변환된 이미지데이터, 정답(라벨)
    #split = train,validation,test
    return beans[split].remove_columns([c for c in beans[split].column_names if c not in cols]) # train,val,test에 없는 컬럼명을 지움->메모리 절약
    # 모든 컬럼이름 목록을 하나씩 반복하면서 리스트 컴프리핸션을 활용하여 cols의 없는 컬럼들만 선택해서 선택된 컬럼들을 삭제함
    # 필요한 컬럼만 남기고 나머지는 삭제하여 메모리 아낌 
train,val,test = keep("train"),keep("validation"),keep("test")

# 평가지표
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def metrics(p):
    predictions, labels = p
    preds = predictions.argmax(-1) # 각 샘플마다 가장 높은 점수를 가진 클래스 선택 (마지막 차원 = 확률차원 중 가장 큰 값의 인덱스 선택)

    return {
        "acc": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"]
    }

# 학습 설정
# %pwd : present working directory(현재 작업 폴더) 확인가능
# TrainingArguments 설정: 학습 스케쥴 WBS
args = TrainingArguments( 
    output_dir="/content/vit_beans_2025",
    eval_strategy="epoch",         
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3, # 15~30 : 실무, earlystopping활용->100개 이상이어도 상관 x
    learning_rate=2e-5,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(), 
    seed=2025,
    remove_unused_columns = False # 이미지 데이터셋 컬럼 유지를 위해 권장
)

trainer = Trainer( # Trainer 초기화 : 학습 스케쥴대로 자동화
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=val,
    # tokenizer=processor, : tokenizer=processor 대신 data_collator를 명시하는 것이 이미지 모델 정석
    data_collator=DefaultDataCollator,
    compute_metrics=metrics
)

# 학습 시작
trainer.train()

# 테스트 데이터 평가
print(trainer.evaluate(test))

# 예측 테스트
for i in [0, 1]:
    ex = beans["test"][i]
    input_tensor = ex["pixel_values"].clone().detach() 
    # clone : 복사본을 만들어 안전하게 처리 # detach: gradient 연결 끊기
    inputs = input_tensor.unsqueeze(0).to(model.device) # 배치차원추가,GPU이동
    with torch.no_grad():
        logits = model(inputs).logits
        pred = logits.argmax(-1).item() # 가장 큰 값의 인덱스 번호 반환

label_key = "labels" if "labels" in ex else "lable"
true_label_id = ex[label_key].item() # 핵심 수정: .item()을 붙여서 tensor(0) -> 0 (정수)으로 변환

# 결과 출력
print(f"[{i}] 예측: {model.config.id2label[pred]} | 정답: {model.config.id2label[true_label_id]}")