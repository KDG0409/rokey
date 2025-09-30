#리스트 데이터 스왑(함수):얕은 복사 이용
#id():메모리 주소 확인

def funca(cb): #매개변수로 인수 얕은 복사
    temp = cb[0]
    cb[0] = cb[1]
    cb[1] = temp

ca = [ 10 , 11 ]
print(ca)
print(id(ca[0]),id(ca[1]))

funca(ca)
print(ca)
print(id(ca[0]),id(ca[1]))