class Cadd:
    def fadd(self,a=1,b=2):
        self.x = a
        self.y = b
        self.hap = self.x +self.y      

ohap = Cadd()
ohap.fadd(10,20)
print(ohap.hap)
ohap.fadd(10)
print(ohap.hap)
ohap.fadd()
print(ohap.hap)
