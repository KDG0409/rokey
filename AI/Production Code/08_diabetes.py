# 당뇨병 예측 시스템(실무형)

# 전체 ML 파이프라인 구축
# DataLoader를 사용한 배치 학습
# Early Stopping 구현
# Learning Rate Scheduler 활용
# 모델 저장 및 로드
# 실무 코드 구조

# 기본 설정
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json

np.random.seed(42)
torch.manual_seed(42)

plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True

class Config:
    def __init__(self):
        self.test_size = 0.2
        self.val_size = 0.2
        self.random_state = 42
        self.input_dim = 10
        self.hidden_dims = [64, 32, 16]
        self.dropout_rate = 0.3
        self.batch_size = 32
        self.num_epochs = 200
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.patience = 20
        self.min_delta = 0.001
        self.scheduler_step_size = 30
        self.scheduler_gamma = 0.5

config = Config()
print('Config 생성 완료!')
print(f'Batch Size: {config.batch_size}')
print(f'Learning Rate: {config.learning_rate}')

# 데이터 처리 클래스 정의 (정규화->분류->배치) 및 텐서 데이터 생성
# 데이터 > 텐서 데이터셋 > 데이터 로더 
class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()

    def load_and_prepare_data(self):
        diabetes = load_diabetes()
        X = diabetes.data
        y_regression = diabetes.target
        median = np.median(y_regression)
        y = (y_regression > median).astype(int)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.config.test_size, stratify=y, random_state=self.config.random_state
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.config.val_size,
            stratify=y_train_val, random_state=self.config.random_state
        )

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val) # val은 fit x
        X_test = self.scaler.transform(X_test) # label은 fit x

        print('데이터 준비 완료')
        print(f'Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}')

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).view(-1, 1) # 파이토치는 2차원 구조 필요 (배치크기,특성)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val).view(-1, 1)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test).view(-1, 1)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        print(f'DataLoader 생성 완료 (Batch Size: {self.config.batch_size})')

        return train_loader, val_loader, test_loader

print('DataPreprocessor 클래스 정의 완료!')

preprocessor = DataPreprocessor(config)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.load_and_prepare_data()
train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
    X_train, X_val, X_test, y_train, y_val, y_test
)

# 예측 클래스 정의 및 예측함수 생성

class DiabetesClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super(DiabetesClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

model = DiabetesClassifier(
    input_dim=config.input_dim,
    hidden_dims=config.hidden_dims,
    dropout_rate=config.dropout_rate
)

print('모델 생성 완료!')

# 얼리스토핑 클래스 정의

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0

print('EarlyStopping 클래스 정의 완료!')

# 학습 클래스 정의

class Trainer:
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device) #예측함수
        self.config = config
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss() #손실함수
        self.optimizer = optim.Adam( #최적화함수
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR( #스케줄러(학습률)
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
        self.early_stopping = EarlyStopping( #얼리스토핑(과도학습방지,메모리 낭비방지)
            patience=config.patience,
            min_delta=config.min_delta
        )
        self.history = { #결과 분석 사용
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }

    def train_epoch(self, train_loader): # 학습 데이터 학습
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            predicted = (outputs >= 0.0).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        return total_loss / total, correct / total

    def validate(self, val_loader): # 검증 데이터에 대입
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item() * batch_X.size(0)
                predicted = (outputs >= 0.0).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader): # 학습>검증>스케줄러>결과도출>얼리스토핑
        print('학습 시작...')

        for epoch in range(self.config.num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1:3d}/{self.config.num_epochs}] "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.6f}")

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"Early Stopping at Epoch {epoch+1}")
                self.model.load_state_dict(self.early_stopping.best_model_state)
                break

        print('\n학습 완료!')

print('Trainer 클래스 정의 완료!')

# 모델 학습 실행

trainer = Trainer(model, config)
trainer.fit(train_loader, val_loader)

# 결과 시각화

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(trainer.history['train_loss'], label='Train Loss')
axes[0, 0].plot(trainer.history['val_loss'], label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Curve')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(trainer.history['train_acc'], label='Train Acc')
axes[0, 1].plot(trainer.history['val_acc'], label='Val Acc')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy Curve')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(trainer.history['learning_rate'], color='orange')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True)

axes[1, 1].plot(trainer.history['train_loss'], label='Train', alpha=0.7)
axes[1, 1].plot(trainer.history['val_loss'], label='Val', alpha=0.7)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Train vs Val Loss')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# 모델 평가

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            preds = (outputs >= 0.0).float()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()

    return all_labels, all_preds, all_probs

all_labels, all_preds, all_probs = evaluate_model(model, test_loader)

print('[Classification Report]')
print(classification_report(all_labels, all_preds, target_names=['Low Risk', 'High Risk']))

cm = confusion_matrix(all_labels, all_preds)
print('\n[Confusion Matrix]')
print(f'              Predicted')
print(f'            Low  High')
print(f'Actual Low  {cm[0,0]:3d}  {cm[0,1]:3d}')
print(f'       High {cm[1,0]:3d}  {cm[1,1]:3d}')

auc = roc_auc_score(all_labels, all_probs)
print(f'\nAUC: {auc:.4f}')

precision, recall, _ = precision_recall_curve(all_labels, all_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

