# 고객 이탈 예측
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
np.random.seed(42)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer  # 열 단위 전처리 파이프라인(매우 유용!!)
from sklearn.pipeline import Pipeline  # 전체 파이프라인 구성 도구
from sklearn.linear_model import LogisticRegression  # 이진 분류용 로지스틱 회귀 모델
from sklearn.metrics import (  # 성능 평가 지표 함수들 임포트
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.inspection import permutation_importance  # 퍼뮤테이션 특성 중요도 계산(매우 유용!!)
import joblib  # 모델 저장/로딩을 위한 직렬화 도구

# 고객 이탈(Churn) "합성 데이터" 생성
# >> 실제 Telco Churn 유사 스키마(수치+범주 혼합)

N = 5000  # 샘플 수
# 수치형 특징 생성: 재직기간(개월), 월요금, 총요금, 고객지원문의 횟수
tenure = np.random.randint(0, 72, size=N)  # 0~71개월 사이 임의 재직기간
monthly_charges = np.round(np.random.normal(60, 15, size=N), 2)
# 평균 60, 표준편차 15의 월요금
monthly_charges = np.clip(monthly_charges, 5, 200)
# 월요금 이상치 클리핑(5~200) >> 정규분포에서 극단적인 값(낮거나 높은 값) 조정
total_charges = np.round(monthly_charges * (tenure + np.random.normal(0.0, 1.0, size=N)), 2)
# 총요금 근사치(월요금 * 재직기간)
support_calls = np.random.poisson(lam=1.8, size=N)  # 고객센터 문의 횟수(포아송 분포)
# 포아송 분포: 일정 기간에 특정사건이 발생하는 횟수을 모델링
# 평균적으로 고객 1명 당 1.8회 정도 문의한다는 기존 통계 정보 활용

# 범주형 특징 생성: 계약형태, 인터넷, 기가옵션, 부가서비스, 결제수단
contract_type = np.random.choice(["month-to-month", "one-year", "two-year"], size=N, p=[0.6, 0.25, 0.15])
# 계약형태p=[0.6, 0.25, 0.15] 월 단위 계약이 60%로 제일 많도록 설정

has_internet = np.random.choice(["yes", "no"], size=N, p=[0.8, 0.2])  # 인터넷 가입 여부
has_giga = np.where((has_internet == "yes") & (np.random.rand(N) < 0.3), "yes", "no")
# 기가옵션(인터넷 가입자 중 일부)
add_on = np.random.choice(["none", "security", "streaming", "both"], size=N, p=[0.4, 0.25, 0.25, 0.10])
# 부가서비스
payment_method = np.random.choice(["credit_card", "bank_transfer", "e_check", "cash"], size=N, p=[0.35, 0.25, 0.30, 0.10])
# 결제수단

# 타깃(이탈 여부) 생성 규칙: 특정 패턴에 의해 확률적으로 이탈 발생
# - 단기 계약(month-to-month), 높은 월요금, 잦은 고객센터 문의, e_check 사용, 인터넷 없음/품질 이슈 등은 이탈 확률을 높임
logit = (
    -2.0  # 기준 절편(이탈 기본 난이도) bias
    + 0.03 * (70 - tenure)  # 재직기간이 짧을수록 이탈↑
    + 0.015 * (monthly_charges - 60)  # 월요금이 높을수록 이탈↑
    + 0.25 * support_calls  # 고객센터 문의 많을수록 이탈↑
    + np.where(contract_type == "month-to-month", 0.8, 0.0)  # 단기 계약 이탈↑
    + np.where(payment_method == "e_check", 0.5, 0.0)  # e_check 이탈↑ (가상의 가정)
    + np.where(has_internet == "no", 0.3, 0.0)  # 인터넷 미가입자는 서비스 가치↓ → 이탈↑
    + np.where((has_internet == "yes") & (has_giga == "no") & (monthly_charges > 80), 0.25, 0.0)
    # 고요금+기가 미사용 이탈↑
)
prob = 1 / (1 + np.exp(-logit))  # 로지스틱 변환으로 이탈 확률 계산()
churn = np.random.binomial(1, prob, size=N)  # 베르누이 샘플링으로 0/1 타깃 생성 : 이항분포

# 데이터프레임 구성
df = pd.DataFrame({  # pandas DataFrame으로 합치기
    "tenure": tenure,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "support_calls": support_calls,
    "contract_type": contract_type,
    "has_internet": has_internet,
    "has_giga": has_giga,
    "add_on": add_on,
    "payment_method": payment_method,
    "churn": churn
})

# 학습/평가 데이터 분할 (Stratify로 타깃 비율 유지)

X = df.drop(columns=["churn"])  # 피처(입력 변수)만 분리
y = df["churn"]  # 타깃(이탈 여부) 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 타깃 비율 유지(stratify)
)

# 전처리 파이프라인 구성
#    - 수치형: 표준화(StandardScaler)
#    - 범주형: 원-핫 인코딩(OneHotEncoder)
#    - 모델: LogisticRegression(class_weight="balanced")

numeric_features = ["tenure", "monthly_charges", "total_charges", "support_calls"]  # 수치형 열 목록
categorical_features = ["contract_type", "has_internet", "has_giga", "add_on", "payment_method"]  # 범주형 열 목록

numeric_transformer = Pipeline(steps=[  # 수치형 전처리 파이프라인
    ("scaler", StandardScaler())  # 표준화로 스케일 맞추기
])

categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # 미지의 범주 무시, 밀집배열 출력

preprocessor = ColumnTransformer(  # 열 단위 전처리기 결합(매우 중요)
    transformers=[
        ("num", numeric_transformer, numeric_features),  # 수치형 처리
        ("cat", categorical_transformer, categorical_features)  # 범주형 처리
    ],
    remainder="drop"  # 지정되지 않은 열은 제거
    # remainder="passthrough"  # 그대로 전달(유지)
)

clf = LogisticRegression(  # 분류 모델 정의
    class_weight="balanced",  # 이탈/비이탈 불균형을 자동 가중치로 보정:소수클래스(이탈)에 큰 가중치 부여
    max_iter=200,  # 수렴 여유 증가
    solver="liblinear",  # 소규모/희소 특성에 안정적인 솔버(소규모 데이터셋에 적합)
    random_state=42  # 재현성
)

pipe = Pipeline(steps=[  # 전처리+모델 전체 파이프라인
    ("preprocess", preprocessor),  # 전처리 단계
    ("model", clf)  # 분류 모델 단계
])

# 모델 학습
pipe.fit(X_train, y_train)  

# 기본 임계값(0.5) 평가
y_proba = pipe.predict_proba(X_test)[:, 1]  # 이탈(1) 클래스의 예측 확률
y_pred = (y_proba >= 0.5).astype(int)  # 기본 임계값 0.5로 이진 예측

acc = accuracy_score(y_test, y_pred)  # 정확도 계산
prec = precision_score(y_test, y_pred, zero_division=0)  # 정밀도 계산
rec = recall_score(y_test, y_pred, zero_division=0)  # 재현율 계산
f1 = f1_score(y_test, y_pred, zero_division=0)  # F1 점수 계산
auc = roc_auc_score(y_test, y_proba)  # ROC-AUC 계산

# 혼동행렬 시각화 (matplotlib 기본)
cm = confusion_matrix(y_test, y_pred)  # 혼동행렬 계산
fig, ax = plt.subplots(figsize=(4.5, 4))  # 그림/축 생성
im = ax.imshow(cm, cmap="Blues")  # 행렬 이미지 표시(컬러맵: Blues)
ax.set_title("Confusion Matrix (Threshold=0.5)")  # 제목
ax.set_xlabel("Predicted")  # x축 라벨
ax.set_ylabel("Actual")  # y축 라벨
for (i, j), v in np.ndenumerate(cm):  # 각 셀에 값 표기: 행/열/값
    ax.text(j, i, str(v), ha="center", va="center")  # 중앙 정렬 텍스트
plt.colorbar(im)  # 컬러바 표시
plt.tight_layout()  # 레이아웃 자동 조정
plt.show()  # 그래프 출력

# ROC 곡선 시각화
RocCurveDisplay.from_predictions(y_test, y_proba)  # 예측 기반 ROC 디스플레이
plt.title("ROC Curve")
plt.show()

# 임계값 최적화 (F1 기준으로 베스트 임계값 탐색)

thresholds = np.linspace(0.1, 0.9, 81)  # 0.1~0.9 범위에서 0.01 간격 후보 임계값
best_thr, best_f1 = 0.5, f1  # 초기값(기본 임계값 성능) = 0.5
for thr in thresholds:  # 모든 후보 임계값 순회
    y_hat = (y_proba >= thr).astype(int)  # 임계값 적용 예측
    f1_tmp = f1_score(y_test, y_hat, zero_division=0)  # F1 계산
    if f1_tmp > best_f1:  # 더 좋은 F1이면 갱신
        best_f1 = f1_tmp  # 최고 F1 갱신
        best_thr = thr  # 최고 임계값 갱신

# 최적 임계값 재평가
y_pred_opt = (y_proba >= best_thr).astype(int)  # 최적 임계값 적용 예측
acc_opt = accuracy_score(y_test, y_pred_opt)  # 정확도
prec_opt = precision_score(y_test, y_pred_opt, zero_division=0)  # 정밀도
rec_opt = recall_score(y_test, y_pred_opt, zero_division=0)  # 재현율
f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)  # F1

# 해석 가능성: 특성 중요도 파악
#    - 로지스틱 회귀 계수(OneHot 이후 열 이름 정리)
#    - 퍼뮤테이션 중요도(모델 전체 예측 민감도 기반)
# ============================================
# 전처리 후 특성 이름 가져오기
# 주의: OneHotEncoder는 get_feature_names_out로 원-핫 열 이름을 제공

ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]  # 범주형 인코더 추출
num_names = numeric_features  # 수치형 열 이름 리스트
cat_names = list(ohe.get_feature_names_out(categorical_features))  # 범주형 원-핫 열 이름 리스트
feature_names = num_names + cat_names  # 전체 특성 이름 결합

# 회귀계수 추출(절편 제외)
coef = pipe.named_steps["model"].coef_.ravel()  # 1차원 회귀계수 배열 추출
coef_df = pd.DataFrame({  # 계수와 열 이름을 데이터프레임으로 정리
    "feature": feature_names,
    "coef": coef
}).sort_values("coef", key=lambda s: s.abs(), ascending=False).head(15)  # 절댓값 큰 상위 15개 # 내림차순

# 퍼뮤테이션 중요도(검증셋 기준, n_repeats는 시간/안정성 트레이드오프)
# 참고: 파이프라인 전체에 대해 permutation_importance 적용 가능

perm = permutation_importance(  # 퍼뮤테이션 중요도 계산
    pipe, X_test, y_test, n_repeats=5, random_state=42, scoring="f1"  # F1 기준 중요도
)
print(perm)
# 상위 10개 시각화
sorted_idx = perm.importances_mean.argsort()[::-1][:10]  # 평균 중요도 기준 내림차순 상위 10
plt.figure(figsize=(6, 4))  # 그림 크기 지정
plt.bar(range(len(sorted_idx)), perm.importances_mean[sorted_idx])  # 막대 그래프
plt.xticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx], rotation=45, ha="right")  # x축 라벨
plt.title("Permutation Importance (Top 10) — Scoring: F1")  # 제목
plt.tight_layout()  # 레이아웃 조정
plt.show()  # 출력

# 모델 저장/불러오기
# ============================================
joblib.dump(pipe, "churn_model_pipeline.joblib")  # 학습된 파이프라인을 파일로 저장
print("\n모델이 'churn_model_pipeline.joblib' 이름으로 저장되었습니다.")  # 저장 완료 안내

# 저장된 모델 로딩 테스트(옵션)
loaded = joblib.load("churn_model_pipeline.joblib")  # 파일에서 모델 로드
test_example = X_test.iloc[[0]]  # 테스트 샘플 1건 선택
pred_proba_example = loaded.predict_proba(test_example)[:, 1][0]  # 이탈 확률 예측
print(f"불러온 모델 테스트 — 샘플 1건 이탈 확률: {pred_proba_example:.4f}")  # 예측 결과 출력