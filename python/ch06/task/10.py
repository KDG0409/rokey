# 1부터 100까지의 수 중 입력받은 수(1부터9)의 배수의 합

num = int (input( " 1부터 9까지의 정수를 입력하세요. : " ))

total = 0
for n in range( 1 , 101) :
    if n % num == 0 :
        total += n

print(total)
