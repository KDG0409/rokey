#1. c
#2. d
#3/4/5/6/7/8/9.
class Phone:
    def __init__(self, a,b,c):
        print('휴대폰 생성')
        self.a = a
        self.b = b
        self.c = c
    def info(self):
        print(self.a,self.b,self.c)
    def setinfo(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c

my_phone = Phone('apple',2025,'white')
print(my_phone.a,my_phone.b,my_phone.c)
my_phone.info()
my_phone.setinfo('samsung',2025,'black')
my_phone.info()

#10. b