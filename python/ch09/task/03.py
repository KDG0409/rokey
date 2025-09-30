class Cadd:
    z=10
    def fadd(self, a,b):
        self.x=a
        self.y=b
        self.hap=self.x+self.y+self.z

obj=Cadd()
obj.fadd(10,20)
print(obj.x)
print(obj.y)
print(obj.hap)