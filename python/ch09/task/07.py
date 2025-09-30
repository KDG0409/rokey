class Fruit:
    name = '오렌지'
    color = '노란색'
    def taste(self):
        print('새콤하다')

orange = Fruit()
print(orange.name)
print(orange.color)
orange.taste()