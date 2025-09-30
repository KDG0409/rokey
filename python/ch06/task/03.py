# 함수 내의 함수

def funca() :
    print("funca 함수 호출")

def funcb() :
    funca()
    print("funcb 함수 호출")

def funcc() :
    funcb()
    print("funcc 함수 호출")

funcc()
