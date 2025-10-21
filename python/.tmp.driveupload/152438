#제너레이터 (함수정의)

def simple_generator():
    yield 'a'
    yield 'b'
    yield 'c'

g = simple_generator()

print(next(g))
print(next(g))
print(next(g))
print(type(g))

def mygen(): 
    for i in range(1,1000):
        result = i *i
        yield result  #제너레이터

gen = mygen()
print(type(gen))
print(next(gen))
print(next(gen))
print(next(gen))
