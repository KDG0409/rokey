class Phone :
    print("휴대폰 생성")
    def __init__(self,manC,relY,col):
        self.manC = manC
        self.relY = relY
        self.col = col

    def info(self):
        print(my_phone.manC)
        print(my_phone.relY)
        print(my_phone.col)

    def setInfo(self,rmanC,rrelY,rcol):
        self.manC = rmanC
        self.relY = rrelY
        self.col = rcol        

my_phone = Phone("samsung",2025,"black")
my_phone.setInfo("Apple",2025,"white")
my_phone.info()

