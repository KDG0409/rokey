#사칙연산 함수 정의(입력)

def fadd(num1,num2):
    num3 = num1 + num2
    return num3

def fsub(num1,num2):
    num3 = num1 - num2
    return num3

def fmul(num1,num2):
    num3 = num1 * num2
    return num3

def fdiv(num1,num2):
    num3 = num1 / num2
    return num3

num1 = float(input("첫 번째 숫자를 입력하세요.: "))
num2 = float(input("두 번째 숫자를 입력하세요.: "))

addnum = fadd(num1,num2)
subnum = fsub(num1,num2)
mulnum = fmul(num1,num2)
divnum = fdiv(num1,num2)

print ( num1, "과" , num2 , "의 더하기 값은", addnum , "이다." )
print ( num1, "과" , num2 , "의 빼기 값은", subnum , "이다." )
print ( num1, "과" , num2 , "의 곱하기 값은", mulnum , "이다." )
print ( num1, "과" , num2 , "의 나누기 값은", divnum , "이다." )