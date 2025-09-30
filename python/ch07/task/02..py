#네임스페이스 확인

b = 0
print("b의 값은",b) #0
b = 1
print("b의 값은",b) #1

def scope_test():
    a = 1
    print("scope_test() 함수 안의 a 값은",a)

a = 0
print("scope_test() 함수 안의 a 값은",a) #0
scope_test() #1
print("scope_test() 함수 안의 a 값은",a) #0