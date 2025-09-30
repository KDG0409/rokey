# 프로그램 확인하고 결과 예측

for sb in range( 1, 11 ,1 ) :
    total = 0 #total이 0으로 재할당
    total = total + sb 
print(total) #10

total = 0 
for sb in range( 1, 11 ,1 ) :
    total = total + sb  # 1부터 10까지의 합
print(total) #55

total = 0 
for sb in range( 1, 11 ,1 ) :
    total = total + 1  # 1을 10번 더함
print(total) #10

for sb in range( 1, 11 ,1 ) :
    total = 0 #total이 0으로 재할당
    total = total + 1 
print(total) #1

total = 0 
for sb in range( 1, 11 ,1 ) :
    total = total + sb  # 1부터 10까지의 합
print(total, end="   ")
print("끝")