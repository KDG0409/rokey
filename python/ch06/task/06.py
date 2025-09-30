#사칙연산 함수 정의

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

addnum = fadd(100,3)
subnum = fsub(100,3)
mulnum = fmul(100,3)
divnum = fdiv(100,3)

print ("더하기 값은", addnum)
print ("빼기 값은", subnum)
print ("곱하기 값은", mulnum)
print ("나누기 값은", divnum)