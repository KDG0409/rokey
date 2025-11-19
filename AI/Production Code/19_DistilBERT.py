import torch, transformers, datasets, sklearn, evaluate, sys, platform
import random, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import evaluate

SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# 참고자료 링크
# https://huggingface.co/datasets
# https://wikidocs.net/21695
# https://wikidocs.net/31698
# http://w.elnn.kr/search/

# 데이터 전처리
dataset = load_dataset("imdb")