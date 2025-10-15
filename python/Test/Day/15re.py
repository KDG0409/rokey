#1. b
#2. b
#3. b
#4. b
#5. a
#6. 
# numbers = [1, 2, 3, 4, 5]
# num=iter(numbers)
# for n in num:
#     print(n)
#7.
# fruits = ["apple", "banana", "cherry"]
# fruit = iter(fruits)
# for i in range(len(fruits)):
#     try:
#         print(next(fruit))
#     except StopIteration:
#         continue
#8.
# for num in iter(range(0,10)):
#     print(num**2)
#9.
for num in iter(range(0,10)):
    if num % 2 == 0 :
        print(num)
#10.
class MyRange:
    def __init__(self,start,stop,step):
        self.start = start
        self.stop = stop
        self.step = step
        self.position = start
    def __iter__(self):
        return self
    def __next__(self):
        if self.position >= self.stop-self.start:
            raise StopIteration
        result = self.position
        self.position += self.step
        return result
data = MyRange(0,10,1)
for i in MyRange(0,10,1):
    print(i)
