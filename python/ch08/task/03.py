#리스트를 반환하는 프로그램

def fk(cb) :
    total = 0
    for sb in range(0,3,1):
        total += cb[sb]
    cb[2] = total
    return cb

ca = [10,20,30]
print(ca)
cb=fk(ca)
print(ca)
print(cb)