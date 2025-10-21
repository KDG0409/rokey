class Myiter():
    def __init__(self,data):
        self.data = data
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.index > len(self.data) :
            raise StopIteration
        self.result = self.data[self.index] ** 2
        self.index += 1
        return self.result
    
list = range(0,10)
iter_list = Myiter(list)
for i in list:
    print(next(iter_list))

