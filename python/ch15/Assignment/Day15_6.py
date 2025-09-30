class Myiter():
    def __init__(self,start,stop,step):
        self.start = start
        self.stop = stop - 1
        self.step =step
        self.index = start
    def __iter__(self):
        return self
    def __next__(self):
        if self.index > self.stop - self.start :
            raise StopIteration
        self.data = self.index
        self.index += self.step
        return self.data
    

iter_list = Myiter(0,10,2)
for i in iter_list:
    print(i)
