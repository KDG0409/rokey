# 선형 대수 계산(선형 방정식)
from scipy.linalg import solve

A = [[3,1],[1,2]]
b = [9,8]
x = solve(A,b)
print(f"solution:{x}")

# 최적화(최소값 계산)
from scipy.optimize import minimize

def f(x):
    return x**2 + 4*x + 4

result = minimize(f,x0=0)
print(f"Optimal value : {result.x}")

# 최적화(방정식의 해)
from scipy.optimize import root

def equation(x):
    return x**2 - 4

sol = root(equation,x0=0)
print(f"root={sol.x}")

# 통계
from scipy.stats import describe

data = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 ]
stats = describe(data)
print(stats)

# 희소행렬(COO포맷)
from scipy.sparse import csr_matrix

data=[[0,0,3],[4,0,0],[0,5,0]]
sparse_matrix = csr_matrix(data)
print(sparse_matrix)