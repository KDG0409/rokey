# 전역변수가 지역변수 안에서 읽히기는 가능
# 전역변수가 지역변수 안에서 수정은 불가능

x = 10
def fadd(num) :
    b = x + num
    print("변수 x값은",x)
    print("변수 b값은",b)
fadd(10)

x = 10
def fadd(num) :
    # x = 10 # 지역변수 선언하지 않으면 글로벌 변수를 지역내에서 수정불가
    x = x + num # 지역변수 선언 후 산술연산이 우선, x+num의 지역변수 x가 None(초기화전)
    print("변수 x값은",x)

fadd(10)

x = 10
def fadd(num) :
    global x
    x = x + num
    print("변수 x값은",x)
    
fadd(10)
