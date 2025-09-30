def my_func():
    print('토끼야 안녕')

my_func()

def add(num1,num2):
    return num1 +num2

print(add(2,3))

def funca(pa,pb):
    nc = pa+pb #지역 변수
    return nc 
na = 10
nb = 11

nc = funca( na , nb ) #전역 변수 
print( na , "+" , nb , "=" , nc )