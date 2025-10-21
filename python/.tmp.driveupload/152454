# 자동차 클래스 정의

class Car:
    def __init__(self,wheel,price):
        self.wheel = wheel
        self.price = price

car = Car(4,30000)
print(car.wheel)
print(car.price)

#자동차 클래스 상속,오버라이드

class Bicycle(Car): #상속
    def __init__(self,year,wheel,price,drivetrain): #오버라이드
        super().__init__(wheel,price) #super()함수
        self.year = year
        self.drivetrain = drivetrain
    def info(self):
        print(f"year : {self.year}")
        print(f"wheel : {self.wheel}")       
        print(f"price : {self.price}") 
        print(f"drivetrain : {self.drivetrain}")             

bicycle = Bicycle(2021,4,30000,'시마노')
# print(bicycle.year)
# print(bicycle.wheel)
# print(bicycle.price)
# print(bicycle.drivetrain)
bicycle.info()