with open("C:/rokey/python/week03/data.txt",'w',encoding = 'utf-8') as file :
    for i in range(1,11):
        file.write(f"{i}번째 줄입니다.")

print("파일이 성공적으로 작성되었습니다.")

add_data = "11번째 줄입니다."

with open("C:/rokey/python/week03/data.txt",'a',encoding = 'utf-8') as file :
    contents = file.write(add_data)

with open("C:/rokey/python/week03/data.txt",'r',encoding = 'utf-8') as file :
    contents = file.read()

print('파일내용',contents)