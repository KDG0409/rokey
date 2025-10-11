##이터레이터 클래스 -> 객체 생성 후 객체를 for/next 사용하여 호출
# class MyIterator:
#     def __init__(self,data):
#         self.data = data
#         self.position = 0
#     def __iter__(self):
#         return self
#     def __next__(self):
#         if self.position > len(self.data):
#             raise StopIteration
#         result = self.data[self.position]
#         self.position += 1
#         return result
    
#제너레이터 함수
# def myGenerator(data):
#     for i in range(len(data)):
#         yield data[i]

# len = range(0,10,1)
# print(len[0])

#컴프리핸션
# lista = (x**2 for x in range(1,10)if x>5)
# for num in lista:
#     print(num)

#이터레이터로 변환
# food = ["김밥", "만두", "양념치킨", "족발", "피자", "쫄면", "라면"]
# foods = iter(food)
# print(type(foods))

##파일 제너레이터
# with open('C:/rokey/python/Test/Day/계좌1.txt','r',encoding='UTF-8')as f:
#     lines = f.readlines()
#     def mygen(data):
#         for line in lines:
#             yield line
#     ans = mygen(lines)
#     print(next(ans))
#     print(next(ans))
#     print(next(ans))
#     print(next(ans))