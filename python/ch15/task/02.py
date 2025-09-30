# 클래스로 이터레이터 생성
#이터레이터 생성방법: iter(),클래스,generator comprehension 등

class MyIterator:
    def __init__(self,data):
        self.data = data
        self.position = 0 # 이터러블->이터레이터
    def __iter__(self):
        return self
    def __next__(self): # 이터레이터->차례로 값 반환
        if self.position >= len(self.data):
            raise StopIteration
        result = self.data[self.position]
        self.position += 1
        return result
    
if __name__ == "__main__":
    i = MyIterator([1,2,3])
    for data in i:
        print(data)