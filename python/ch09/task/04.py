class Car: #설계도, 틀
    def __init__(self,pnum,pspeed): #클래스 내 변수 초기화 용도
        self.num=pnum
        self.speed=pspeed
    def fprint(self):
        print(self.num)
        print(self.speed)

new_car=Car(2023,90) # 새 자동차 생산 출고
new_car.fprint()
# print(new_car.num)
# print(new_car.speed)
my_car = Car(2022,80) # 내 자동차 생산 출고
my_car.fprint()