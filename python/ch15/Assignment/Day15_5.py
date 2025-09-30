class MyIter():
    def __init__(self,data):
        self.data = data 
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.index > len(self.data):
            raise StopIteration
        if self.data[self.index] % 2 == 0 :
            result = self.data[self.index]
            self.index += 1
            return result
        else :
            self.index += 1
            return ""

iter_list = MyIter(range(11))
for num in range(11) :
    print(next(iter_list))