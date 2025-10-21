## 파일 내용 분류해서 읽기
with open("C:/rokey/python/Test/Day/계좌1.txt",'a',encoding = 'UTF-8') as f:
    f.write("")
    f.write("강호동 147-12-002093\n")
    f.write("유재석 146-22-102093\n")
with open("C:/rokey/python/Test/Day/계좌1.txt",'r',encoding = 'UTF-8') as f:
    data=f.readlines()
    name_list=[]
    add_list=[]
    tlist={}
    for line in data:
        data2 = line.split()
        name_list.append(data2[0])
        add_list.append(data2[1])
        tlist[data2[0]]=data2[1]
    print(add_list)
    print(tlist)
    

    
