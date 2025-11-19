import torch, transformers, datasets, evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import evaluate
from transformers import TrainingArguments, Trainer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
ds = load_dataset("glue", "sst2")

MODEL="distilbert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(MODEL,use_fast=True) # 전처리기

def preprocess(ex): # 전처리 결과 반환
    return tokenizer(ex["sentence"], truncation=True, max_length=256)
enc=ds.map(preprocess, batched=True, remove_columns=["sentence","idx"]) # 매핑
data_collator=DataCollatorWithPadding(tokenizer=tokenizer) # 동적 패딩(정렬)
model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2).to(device) # 사전학습모델

acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def metrics(p):
    predictions, labels = p
    # predictions가 튜플로 나오는 경우(logits 외 다른 요소가 있는 경우)를 대비해 분기 처리하거나
    # 일반적인 경우 predictions 자체가 logits입니다.
    # 여기서는 numpy 배열이라고 가정하고 처리합니다.
    preds = predictions.argmax(-1)

    return {
        "acc": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"]
    }

args = TrainingArguments( # TrainingArguments 설정: 학습 스케쥴 WBS
    output_dir="/content/sst2_2025",
    eval_strategy="epoch",          # 수정됨: evaluation_strategy -> eval_strategy
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(), # GPU가 있을 때만 fp16 사용
    # fp16 : float precision(부동소수점 정밀도) 16bit로 공간할당하여 메모리 절약, 속도 향상
    report_to="none",
    seed=2025
)

trainer = Trainer( # Trainer 초기화 : 학습 스케쥴대로 자동화
    model=model,
    args=args,
    train_dataset=enc["train"],
    eval_dataset=enc["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=metrics
)

txt=["This movie was amazing!","Worst film ever."]
inp=tokenizer(txt,return_tensors="pt",padding=True,truncation=True,max_length=256).to(model.device)
with torch.no_grad(): 
    out=torch.softmax(model(**inp).logits,dim=-1).cpu().numpy() # **inp : 입력인자 언패킹 후 딕셔너리->키워드 인수로 풀어서 전달(파이토치형태)
    # inp = {"id": [].....} >> model(input_id = .....)
    for t,p in zip(txt,out): 
        print(f"{t}\n→ Negative={p[0]:.3f}, Positive={p[1]:.3f}")