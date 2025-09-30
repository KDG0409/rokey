# 나누기 연산 함수 정의

def fdiv(num1,num2) :
    if num2 == 0 :
        print( "0으로는 나눌 수 없다.")
    else :
        num3 = num1 / num2
        print( num1 , "을", num2 , "로 나눈 값은" , num3 , "이다." )
    
n1 = int(input(" 첫 번째 변수를 입력하세요.: "))
n2 = int(input(" 두 번째 변수를 입력하세요.: "))

fdiv(n1,n2)