class Myiter ():
    def __init__(self,data) :
        self.data = data
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.index > len(self.data) :
            raise StopIteration
        self.result = self.data[self.index]
        self.index += 1
        return self.result

        
fruits = ["apple", "banana", "cherry"]
iter_list = Myiter(fruits)
print(next(iter_list))
print(next(iter_list))
print(next(iter_list))