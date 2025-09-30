def gen(num):
    for i in range(1,num+1) :
        yield i*i

n = 10
ans = gen(n)

for i in range(n):
    print(next(ans))