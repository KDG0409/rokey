#1. b
#2. a
#3. b
#4. b
#5/6/7/8/10.
class Phone:
    def __init__(self,number,color):
        self.number=number
        self.color=color
    def showInfo(self):
        print(self.number)
        print(self.color)

class SmartPhone(Phone):
    def __init__(self,number,color,company):
        super().__init__(number,color)
        self.company=company
    def showInfo(self):
        super().showInfo()
        print(self.company)
#9. 해당 클래스의 메서드와 변수를 상속받았기 때문이다.
