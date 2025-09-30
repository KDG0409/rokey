ca = [10,11]
cb = ca
print(ca)
print(cb)
temp = cb[0]
cb[0] = cb[1]
cb[1] = temp
print(ca)
print(cb)

na = 10
nb = 11
def func(na,nb):
    pa=na
    pb=nb
    temp=pa
    pa=pb
    pb=temp
    nc=pa+pb
    return nc
func(na,nb)
print(na,nb)