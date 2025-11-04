import numpy as np
import matplotlib.pyplot as plt
import torch
from torchviz import make_dot
from keras.datasets import mnist
# from IPython.display import display

# tensor (0/1/2/3/4계 tensor)

r0 = torch.tensor(1.0).float()
x4 = torch.FloatTensor([[1, 2],[3, 4]])
float_tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
print(type(r0))
print(r0.dtype)
print(r0.shape)
print(r0.data)

item = r0.item() # item 함수는 0계(스칼라) 또는 요소 수가 하나뿐인 텐서에서 사용 가능

r1_np = np.array([1, 2, 3, 4, 5])
r1 = torch.tensor(r1_np).float()

print('requires_grad: ', r1.requires_grad) # 속성
print('device: ', r1.device)

r2_np = np.array([[1, 5, 6], [4, 3, 2]])
r2 = torch.tensor(r2_np).float()
r2_np = r2.data.numpy() # tensor->넘파이로 변환
r2_np2 = np.array(r2) # list->넘파이로 변환
r2_t = torch.tensor(r2_np) #텐서로 변환

x1 = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

print("size:", x1.size())  # 텐서크기 : [3,3]
print("rank(차원):", x1.dim()) # 차원 (축)의 수 : 2

print(torch.max(r2, 1)[1]) # max 함수의 두 번째 인자는 축을 의미, [0]은 값 [1]은 인덱스를 의미

torch.manual_seed(123)
r3 = torch.randn((3, 2, 2))
tensor_3D = np.arange(24).reshape(2, 3, 4)
print(tensor_3D)

r4 = torch.ones((2, 3, 2, 2))

r5 = r1.long() # 정수 타입 텐서로 변환

r6 = r3.view(3, -1) # reshape 역할/ -1은 자동생성
r7 = r3.view(-1)
r3.view(-1)[-1] # 인덱싱 가능

# 경사 계산
x_np = np.arange(-2, 2.1, 0.25) #-2부터 2.1 미만까지 0.25간격으로
x = torch.tensor(x_np, requires_grad=True, dtype=torch.float32) # requires_grad=True : 자동 경사계산 기능
y = 2 * x**2 + 2
z = y.sum()
z.backward()
x.grad.zero_()

# 시각화 함수 호출
g= make_dot(z, params={'x': x}) # g = make_dot(z, params=params)
# display(g)

# 시그모이드 함수 경사 계산
sigmoid = torch.nn.Sigmoid()
y = sigmoid(x)
z = y.sum()
z.backward() # 역전파
print(x.grad)
x.grad.zero_() # 경사 초기화
plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()

# Tensor 생성/속성
tensor_zeros = torch.zeros(2, 3)
tensor_ones = torch.ones(2, 3)
tensor_full = torch.full((2, 3), 7)
random_tensor = torch.randn((2, 3))
tensor_like = torch.randn_like(random_tensor) # 같은 크기, 같은 형식
tensor_arange = torch.arange(1, 7).reshape(2, 3)

x = torch.randn(3, 4, 5)
print(f"Dimensions: {x.dim()}") # 차원수 : 3 /차원 확인 dim(), ndimension()
print("rank(차원):", x.ndimension())
print(f"Shape: {x.shape}") # 각 차원의 크기를 튜플 형태로 반환 : torch.Size([3, 4, 5])
print(f"Size: {x.size()}") # 각 차원의 크기를 튜플 형태로 반환 : torch.Size([3, 4, 5])
print(f"Data type: {x.dtype}") # 텐서의 데이터 타입을 반환 : 기본은 torch.float32
print(f"Device: {x.device}") # 텐서가 저장된 장치(CPU 또는 GPU) : 기본은 cpu
print(f"Number of elements: {x.numel()}") # 총 원소 개수를 반환 :3*4*5=60

# 행렬 계산
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[7, 8], [9, 10], [11, 12]])
c1 = torch.matmul(a, b) # 행렬곱(외적)
c2 = torch.mul(a, b) # 요소별 곱셈
d = torch.rand(3, 3).reshape(1,9).T # 전치행렬
transposed = d.t() #전치행렬
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(a+b,a-b) # 합,차
dot_product = torch.dot(a, b) #내적
result = a / b # 요소별 나누기
result = torch.div(a, b) 
result = a ** 2 # 요소별 거듭제곱
result = torch.pow(a, 2)
print(result  > 3) # 비교 연산(True,False사용)
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
sum_value_dim0 = torch.sum(tensor, dim=0) # 0(행기준/열방향/세로)
sum_value_dim1 = torch.sum(tensor, dim=1) # 1(열기준/행방향/가로)
max_value = torch.max(tensor, dim=0)
min_value = torch.min(tensor, dim=0)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.rand(3, 3).to(device)

# 텐서 차원 변환
x = torch.tensor([[[1],[2],[3]]]) # ([1, 3, 1])
x1_tensor = x.view(2, 3) # 텐서 원소 총 개수 동일
print(x.squeeze()) # 크기가 1인 차원함수 제거 -> ([3])
print(x.squeeze(dim=0)) # 특정차원제거(0번 인덱스)
print(x.unsqueeze(0)) # 맨 앞에 크기가 1인 새로운 차원 추가
print(x.unsqueeze(-1)) # 맨 뒤에 크기가 1인 새로운 차원 추가
print(torch.randint(0, 4, (3,2)).unsqueeze(0).expand(4, 3, 2)) # 확장된 차원을 반복하여 늘림, 주소 공유
rep_res_data = torch.randint(0, 4, (3,2)).unsqueeze(0).repeat(4, 1, 1) # 차원별 반복횟수, 독립 주소

torch.tensor([[1, 2, 3, 4, 5]])
torch.tensor([[1, 2, 3, 4, 5]]).unsqueeze(dim=0) # tensor([[[1, 2, 3, 4, 5]]])

# 텐서 추출(슬라이싱)
slice_1 = tensor[0, :] # 첫 번째 행
slice_2 = tensor[:, 1] # 두 번째 열
slice_3 = tensor[1, 2] # 두 번째 행과 세 번째 열의 원소
slice_4 = tensor[0, 0:2] # 첫 번째 행의 첫 번째와 두 번째 열

# 텐서 분할
chunks = torch.chunk(tensor, 3) # 3 덩어리로 분할, 마지막 덩어리가 지정된 크기보다 작을 수 있음
splits = torch.split(tensor, 2) # 2개의 원소를 가진 덩어리들로 분할, 마지막 덩어리가 지정된 크기보다 작을 수 있음

# 브로드 캐스팅
a = torch.tensor([[1, 2],[3, 5]]) # 2x2 크기의 2차원 텐서 생성
b = torch.tensor([1, 2]) # 1차원 텐서 생성
c = a * b # b가 자동적으로 확장(복제)