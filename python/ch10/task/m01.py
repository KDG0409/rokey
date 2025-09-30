class Cvalue:
    def __init__(self):
        self.lista = []
    def add(self,num):
        self.lista.append(num)
    def fprint(self):
        print(self.lista)

def plus(a,b):
    c = a + b
    return c

p1 = Cvalue()
p1.add(1)
p1.add(2)
p1.add(3)
p1.fprint()