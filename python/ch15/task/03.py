# 클래스로 리버스 이터레이터 생성

class ReverseIterator:
    def __init__(self,data):
        self.data = data
        self.position = len(self.data)-1
    def __iter__(self): # 이터러블->이터레이터
        return self
    def __next__(self): # 이터레이터->차례로 값 반환
        if self.position < 0 :
            raise StopIteration
        result = self.data[self.position]
        self.position -= 1
        print(type(self.data))
        return result
    
if __name__ == "__main__":
    i = ReverseIterator([1,2,3])
    for data in i:
        print(data)
    print(type(i))