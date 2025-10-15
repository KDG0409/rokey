##이터레이터 클래스
# class MyIter:
#     def __init__(self,data):
#         self.data=data
#         self.position=0
#     def __iter__(self):
#         return self
#     def __next__(self):
#         if self.position >= len(self.data):
#             raise StopIteration
#         result = self.data[self.position]
#         position += 1
#         return result

#제너레이터 함수

# def Mygen(data):
#     for i in range(len(data)):
#         result=data[i]       
#         yield result

#컴프리핸션

# a=(x**2 for x in range(100) if x%3==0)
# print(next(a))
# print(next(a))

#이터레이터로 변환
# a=range(100)
# b=iter(a)
# # print(list(b))
# print(next(b))
# print(next(b))
# print(next(b))

##파일 제너레이터

# with open('C:/rokey/python/Test/Day/계좌1.txt','r',encoding='UTF-8')as f:
#     lines=f.readlines()
#     def myGen(lines):
#         for line in lines:
#             yield line
#     a = myGen(lines)
#     print(next(a))
#     print(next(a))
#     print(next(a))