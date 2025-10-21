#제너레이터 표현식

gen = (i*i for i in range(1,100))
print(type(gen))
print(next(gen))
print(next(gen))
print(next(gen))

#컴프리핸션
numbers = [1,2,3,4]

lista = [ x**2 for x in numbers if x % 2 == 0 ]
print(type(lista))

lista = { x**2 : x for x in numbers if x % 2 == 0 }
print(type(lista))

lista = ( x**2 for x in numbers if x % 2 == 0 )
print(type(lista))