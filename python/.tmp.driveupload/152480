class Phone :
    def __init__(self,number,color) :
        self.number = number
        self.color = color
    def showInfo(self):
        print("실행결과:")
        print(f"전화번호: {self.number} ")
        print(f"색상: {self.color} ")

class SmartPhone(Phone):
    def __init__(self,number,color,company) :
        super().__init__(number,color)
        self.company = company 

apple = SmartPhone("010-1234-5678", "화이트")
apple.showInfo()
