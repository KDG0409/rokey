import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.0]])  # 배치=1, 클래스=3
target = torch.tensor([0])  # 정답: 고양이(인덱스 0)

# softmax

probs = F.softmax(logits, dim=1)     # 확률값
p_true = probs[0, target]            # 정답 클래스 확률만 뽑기
loss_manual = -torch.log(p_true)     # NLL = -log(p_true)

# log_softmax

log_probs = F.log_softmax(logits, dim=1)  # 로그확률(수치안정)
nll = nn.NLLLoss(reduction='mean')        # 배치 평균(기본)
loss_nll = nll(log_probs, target)

# CrossEntropyLoss = log_softmax + NLLLoss

ce = nn.CrossEntropyLoss(reduction='mean')
loss_ce = ce(logits, target)

# 1. 큰 로짓(overflow 위험한 예)
big_logits = torch.tensor([[1000.0, 999.0, 995.0]])
target = torch.tensor([0])

#  위험: softmax → log (오버플로우/NaN 가능)
try:
    probs_bad = F.softmax(big_logits, dim=1)  # e^1000 계산 중 터질 수 있음
    loss_bad = -torch.log(probs_bad[0, target])
    print("Naive NLL with big logits =", loss_bad.item())
except Exception as e:
    print("Naive 방식에서 문제 발생:", e)

# 안전: log_softmax → NLLLoss (내부에 log-sum-exp trick)
log_probs_safe = F.log_softmax(big_logits, dim=1)
loss_safe = nn.NLLLoss()(log_probs_safe, target)
print("안전한 NLL with big logits =", loss_safe.item())

# 더 간단: CrossEntropyLoss
loss_ce_safe = nn.CrossEntropyLoss()(big_logits, target)
print("안전한 CE with big logits =", loss_ce_safe.item())

# 2. KL Divergence >> 확률분포 간 거리
# 각 클래스의 확률 차이를 모두 더해서 전체적인 차이 확인, 배치 크기에 상관없이 일정한 손실 규모 유지
import torch
import torch.nn.functional as F

P = torch.tensor([1.0, 0.0, 0.0])
Q = torch.tensor([0.7, 0.2, 0.1])
log_Q = torch.log(Q)
kl_div = F.kl_div(log_Q, P, reduction='sum')