class Country:
    name = '미입력'
    population = '미입력'

    def show(self):
        print("부모 클래스 출력입니다.")

class Korea(Country):
    def __init__(self,name,population):
        self.name = name
        self.population = population

    def show(self):
        print("자식 클래스 출력입니다.")

    def show_name(self):
        print("여기는",self.name,"입니다.")

    def show_pop(self):
        print("인구는",self.population,"명 입니다.")

a = Korea('대한민국',"5000만명")
a.show()
a.show_name()
a.show_pop()