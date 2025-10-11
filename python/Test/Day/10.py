# 클래스 상속
# class Car:
#     def __init__(self,a,b):
#         self.wheel = a
#         self.price = b
# class Bicycle(Car):
#     def __init__(self,a,b,c,d):
#         super().__init__(b,c)
#         self.year = a
#         self.drivetrain = d
#     def info(self):
#         print(self.year)
#         print(self.wheel)
#         print(self.price)
# car = Car(4,30000)
# bicycle = Bicycle(2021,2,100,"시마노")
# bicycle.info()
# print(bicycle.drivetrain)

#1. b
#2. a
#3. b
#4. b
#5/6/7/8/10. 
# class Phone:
#     def __init__(self,number,color):
#         self.number = number
#         self.color = color
#     def showInfo(self):
#         print(self.number)
#         print(self.color)
# class SmartPhone(Phone):
#     def __init__(self,number,color,company):
#         super().__init__(number,color)
#         self.company = company
#     def showInfo(self):
#         super().showInfo()
#         print(self.company)
#9. 스마트폰 클래스는 상속을 받아 폰 클래스의 showInfo()매서드 호출가능
