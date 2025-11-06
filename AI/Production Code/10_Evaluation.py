# 혼동행렬 해석
# Precision, Recall, F1 (클래스별/마이크로/매크로)
# ROC-AUC (One-vs-Rest)
# Calibration & 온도 스케일링
# 클래스 불균형 처리


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize # 이진화라벨링
from collections import Counter

# 재현성 설정
torch.manual_seed(42)
np.random.seed(42)

# 불균형 데이터셋

class ImbalancedDataset(Dataset):
    def __init__(self, n_samples=2000, n_features=20, n_classes=4,
                 imbalance_ratio=[0.5, 0.3, 0.15, 0.05]):
        self.n_classes = n_classes
        samples_per_class = [int(n_samples * ratio) for ratio in imbalance_ratio]

        X_list = []
        y_list = []

        for class_idx in range(n_classes):
            n = samples_per_class[class_idx]
            mean = np.random.randn(n_features) * (class_idx + 1)
            cov = np.eye(n_features) * (0.5 + class_idx * 0.2)
            X_class = np.random.multivariate_normal(mean, cov, n)
            y_class = np.full(n, class_idx)

            X_list.append(X_class)
            y_list.append(y_class)

        self.X = torch.FloatTensor(np.vstack(X_list))
        self.y = torch.LongTensor(np.hstack(y_list))

        class_counts = Counter(self.y.numpy())
        print(f"\n전체 클래스 분포: {dict(sorted(class_counts.items()))}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# 모델 정의

class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, n_classes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        return self.network(x)
    
# 온도 스케일링(확률보정)

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, model, val_loader, device, max_iter=50):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        logits_list = []
        labels_list = []

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        print(f"온도 스케일링 완료: T = {self.temperature.item():.3f}")
        return self.temperature.item()
    
# 혼동행렬 시각화

def plot_confusion_matrix_with_analysis(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 혼동행렬
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title('혼동행렬 (Confusion Matrix)', fontsize=14, pad=10)
    axes[0].set_ylabel('실제 클래스', fontsize=11)
    axes[0].set_xlabel('예측 클래스', fontsize=11)

    # 오류 분석
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    error_analysis = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                error_analysis.append({
                    'True': class_names[i],
                    'Pred': class_names[j],
                    'Count': cm[i, j],
                    'Rate': cm_normalized[i, j]
                })

    if len(error_analysis) > 0:
        error_analysis.sort(key=lambda x: x['Rate'], reverse=True)
        top_errors = error_analysis[:min(5, len(error_analysis))]
        error_labels = [f"{e['True']}→{e['Pred']}" for e in top_errors]
        error_rates = [e['Rate'] * 100 for e in top_errors]

        axes[1].barh(error_labels, error_rates, color='coral')
        axes[1].set_xlabel('오류율 (%)', fontsize=11)
        axes[1].set_title('주요 오류 유형', fontsize=14, pad=10)
        axes[1].grid(axis='x', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, '완벽한 분류!',
                    transform=axes[1].transAxes, ha='center', va='center',
                    fontsize=16)
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    return cm

# 평가지표

def calculate_detailed_metrics(y_true, y_pred, y_proba, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics_avg = {
        'macro': precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[:3],
        'micro': precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)[:3],
        'weighted': precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[:3]
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (1) 클래스별 지표
    x = np.arange(len(class_names))
    width = 0.25

    axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[0, 0].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[0, 0].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    axes[0, 0].set_xlabel('클래스', fontsize=11)
    axes[0, 0].set_ylabel('점수', fontsize=11)
    axes[0, 0].set_title('클래스별 평가지표', fontsize=13, pad=10)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(class_names)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, 1.1])

    # (2) 평균 방식 비교
    avg_types = ['Macro', 'Micro', 'Weighted']
    avg_precisions = [metrics_avg['macro'][0], metrics_avg['micro'][0], metrics_avg['weighted'][0]]
    avg_recalls = [metrics_avg['macro'][1], metrics_avg['micro'][1], metrics_avg['weighted'][1]]
    avg_f1s = [metrics_avg['macro'][2], metrics_avg['micro'][2], metrics_avg['weighted'][2]]

    x_avg = np.arange(len(avg_types))
    axes[0, 1].bar(x_avg - width, avg_precisions, width, label='Precision', alpha=0.8)
    axes[0, 1].bar(x_avg, avg_recalls, width, label='Recall', alpha=0.8)
    axes[0, 1].bar(x_avg + width, avg_f1s, width, label='F1-Score', alpha=0.8)
    axes[0, 1].set_xlabel('평균 방식', fontsize=11)
    axes[0, 1].set_ylabel('점수', fontsize=11)
    axes[0, 1].set_title('평균 방식별 비교', fontsize=13, pad=10)
    axes[0, 1].set_xticks(x_avg)
    axes[0, 1].set_xticklabels(avg_types)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_ylim([0, 1.1])

    # (3) Support vs F1
    axes[1, 0].scatter(support, f1, s=100, alpha=0.6, c=range(len(class_names)), cmap='viridis')
    for i, name in enumerate(class_names):
        axes[1, 0].annotate(name, (support[i], f1[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1, 0].set_xlabel('샘플 수', fontsize=11)
    axes[1, 0].set_ylabel('F1-Score', fontsize=11)
    axes[1, 0].set_title('클래스 불균형과 성능', fontsize=13, pad=10)
    axes[1, 0].grid(alpha=0.3)

    # (4) 설명
    explanation = (
        "  평균 방식:\n\n"
        "• Macro: 클래스별 평균\n"
        "  모든 클래스 동등\n\n"
        "• Micro: 전체 샘플 기준\n"
        "  다수 클래스 영향\n\n"
        "• Weighted: 가중 평균\n"
        "  실제 분포 반영"
    )
    axes[1, 1].text(0.1, 0.5, explanation, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    print("분류 리포트")
    print(classification_report(y_true, y_pred, target_names=class_names,
                               digits=3, zero_division=0))

    return metrics_avg

# ROC-AUC

def plot_roc_curves_multiclass(y_true, y_proba, class_names):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 클래스별 ROC
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in enumerate(colors):
        axes[0].plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].set_title('클래스별 ROC (One-vs-Rest)', fontsize=13, pad=10)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.3)

    # Macro/Micro
    axes[1].plot(fpr["micro"], tpr["micro"],
                label=f'Micro-avg (AUC={roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=3)
    axes[1].plot(fpr["macro"], tpr["macro"],
                label=f'Macro-avg (AUC={roc_auc["macro"]:.3f})',
                color='navy', linestyle=':', linewidth=3)
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1].set_title('평균 ROC', fontsize=13, pad=10)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("ROC-AUC 점수")
    print("="*60)
    for i, name in enumerate(class_names):
        print(f"{name:15s}: {roc_auc[i]:.4f}")
    print(f"{'Micro-avg':15s}: {roc_auc['micro']:.4f}")
    print(f"{'Macro-avg':15s}: {roc_auc['macro']:.4f}")

    return roc_auc

# 유틸리티 함수들 (IndexError 완전 해결)

def get_labels_from_loader(loader):
    """DataLoader에서 모든 레이블 추출"""
    labels = []
    for _, batch_labels in loader:
        labels.extend(batch_labels.numpy().tolist())
    return labels

def get_class_weights(train_loader, n_classes):
    """클래스 가중치 계산"""
    labels = get_labels_from_loader(train_loader)
    class_counts = Counter(labels)
    n_samples = len(labels)

    weights = torch.FloatTensor([
        n_samples / (n_classes * class_counts.get(i, 1))
        for i in range(n_classes)
    ])

    print(f"\n학습 데이터 클래스 분포: {dict(sorted(class_counts.items()))}")
    print(f"클래스 가중치: {weights.numpy()}")
    return weights

def create_weighted_sampler(train_loader):
    """Oversampling을 위한 Sampler"""
    labels = get_labels_from_loader(train_loader)
    class_counts = Counter(labels)

    sample_weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

class AugmentedWrapper(Dataset):
    """데이터 증강 Wrapper"""
    def __init__(self, dataset, augment_ratio=0.5, noise_std=0.15):
        self.dataset = dataset
        self.base_len = len(dataset)
        self.augment_len = int(self.base_len * augment_ratio)
        self.total_len = self.base_len + self.augment_len
        self.noise_std = noise_std

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.base_len:
            return self.dataset[idx]
        else:
            # 증강 샘플
            base_idx = (idx - self.base_len) % self.base_len
            x, y = self.dataset[base_idx]
            noise = torch.randn_like(x) * self.noise_std
            return x + noise, y
        
# 학습 및 평가

def train_model(model, train_loader, criterion, optimizer, device, epochs=30):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# 메인 실행

def main():
    print("="*70)
    print("10차수: 고급 평가지표 및 클래스 불균형 처리")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n디바이스: {device}")

    # 데이터 생성
    full_dataset = ImbalancedDataset(
        n_samples=2000, n_features=20, n_classes=4,
        imbalance_ratio=[0.5, 0.3, 0.15, 0.05]
    )

    # 분할
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Weighted Loss 가중손실

    train_loader_normal = DataLoader(train_dataset, batch_size=64, shuffle=True)
    class_weights = get_class_weights(train_loader_normal, n_classes=4).to(device)

    model1 = MultiClassClassifier(20, 64, 4).to(device)
    criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)

    print("\n[학습 시작]")
    train_model(model1, train_loader_normal, criterion_weighted, optimizer1, device, 30)

    print("\n[평가]")
    y_true, y_pred, y_proba = evaluate_model(model1, test_loader, device)

    print("\n1. 혼동행렬")
    plot_confusion_matrix_with_analysis(y_true, y_pred, class_names)

    print("\n2. 상세 지표")
    calculate_detailed_metrics(y_true, y_pred, y_proba, class_names)

    print("\n3. ROC-AUC")
    plot_roc_curves_multiclass(y_true, y_proba, class_names)

    # Oversampling 
    temp_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    sampler = create_weighted_sampler(temp_loader)
    train_loader_oversampled = DataLoader(train_dataset, batch_size=64, sampler=sampler)

    model2 = MultiClassClassifier(20, 64, 4).to(device)
    criterion_normal = nn.CrossEntropyLoss()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

    print("\n[학습 시작]")
    train_model(model2, train_loader_oversampled, criterion_normal, optimizer2, device, 30)

    print("\n[평가]")
    y_true2, y_pred2, y_proba2 = evaluate_model(model2, test_loader, device)

    print("\n혼동행렬")
    plot_confusion_matrix_with_analysis(y_true2, y_pred2, class_names)

    # Data Augmentation 데이터 증강
    augmented_dataset = AugmentedWrapper(train_dataset, augment_ratio=0.5, noise_std=0.15)
    train_loader_augmented = DataLoader(augmented_dataset, batch_size=64, shuffle=True)

    model3 = MultiClassClassifier(20, 64, 4).to(device)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)

    print(f"\n증강 후 데이터 크기: {len(augmented_dataset)} (원본: {len(train_dataset)})")
    print("\n[학습 시작]")
    train_model(model3, train_loader_augmented, criterion_normal, optimizer3, device, 30)

    print("\n[평가]")
    y_true3, y_pred3, y_proba3 = evaluate_model(model3, test_loader, device)

    print("\n혼동행렬")
    plot_confusion_matrix_with_analysis(y_true3, y_pred3, class_names)

    # Temperature Scaling
    temp_scaler = TemperatureScaling().to(device)
    temperature = temp_scaler.calibrate(model1, test_loader, device)

    model1.eval()
    all_probs_before = []
    all_probs_after = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model1(inputs)

            probs_before = F.softmax(logits, dim=1)
            probs_after = F.softmax(temp_scaler(logits), dim=1)

            all_probs_before.extend(probs_before.cpu().numpy().tolist())
            all_probs_after.extend(probs_after.cpu().numpy().tolist())

    all_probs_before = np.array(all_probs_before)
    all_probs_after = np.array(all_probs_after)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    max_probs_before = all_probs_before.max(axis=1)
    max_probs_after = all_probs_after.max(axis=1)

    axes[0].hist(max_probs_before, bins=20, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('최대 확률', fontsize=11)
    axes[0].set_ylabel('빈도', fontsize=11)
    axes[0].set_title(f'보정 전 (평균: {max_probs_before.mean():.3f})', fontsize=13)
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].hist(max_probs_after, bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('최대 확률', fontsize=11)
    axes[1].set_ylabel('빈도', fontsize=11)
    axes[1].set_title(f'보정 후 (평균: {max_probs_after.mean():.3f}, T={temperature:.3f})', fontsize=13)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 평가 비교
    _, _, f1_1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, f1_2, _ = precision_recall_fscore_support(y_true2, y_pred2, average='macro', zero_division=0)
    _, _, f1_3, _ = precision_recall_fscore_support(y_true3, y_pred3, average='macro', zero_division=0)

    methods = ['Weighted Loss', 'Oversampling', 'Augmentation']
    scores = [f1_1, f1_2, f1_3]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, scores, color=['skyblue', 'lightcoral', 'lightgreen'],
                  edgecolor='black', alpha=0.8)

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Macro F1-Score', fontsize=12)
    ax.set_title('불균형 처리 방법별 성능', fontsize=14, pad=15)
    ax.set_ylim([0, max(scores) * 1.2])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n최종 비교:")
    for method, score in zip(methods, scores):
        print(f"  {method:20s}: {score:.4f}")

    print("\n" + "="*70)
    print("강의 완료! 🎓")
    print("="*70)

if __name__ == "__main__":
    main()