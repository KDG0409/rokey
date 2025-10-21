# nums = [1, 2, 3]
# it = iter(nums)
# print(next(it))
# print(next(it))

# def my_gen():
#     yield 1
#     yield 2
#     yield 3
# gen = my_gen()
# print(next(gen))
# print(next(gen))

def countdown(n):
    while n > 0:
        yield n
        n -= 1
gen = countdown(3)
for x in gen:
    print(x, end=" ")
