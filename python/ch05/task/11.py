# 첫 번째 수와 두 번째 수를 입력받아 3의 배수의 총합을 구하는 프로그램 
# for 문과 while 문으로 작성

num1 = int(input("계산할 첫 번째 수를 입력하세요. : "))
num2 = int(input("계산할 마지막 수를 입력하세요. : "))

total = 0

for num in range( num1, num2 +1 ) :
    if num % 3 == 0 :
        total = total + num

print( num1 , "부터" , num2 , "까지 3의 배수의 총합은" , total , "입니다.")

total = 0
i = 0

while i <= num2 :
    if i < num1 :
        i = i + 1
    elif i % 3 == 0 :
        total += i
        i = i + 1
    else :
        i = i + 1

print( num1 , "부터" , num2 , "까지 3의 배수의 총합은" , total , "입니다.")