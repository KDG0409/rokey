#재귀함수 연습

def count_down(n):
    if n == 0 :
        print("완료")
        return
    print(n)
    count_down(n-1)

count_down(5)

# 팩토리얼 계산 

def fac(n) :
    if n == 1 :
        return 1
    else:
        return n * fac(n-1)
    
#fac(n)=n*(n-1)*(n-2)*(n-3)...1 <- 1에서부터 곱해짐

i = int(input(" 팩토리얼 인수 입력: "))
ans = fac(i)
print(ans)