#1. a
#2. a
#3. a
#4. True
#5. b
#6/7/8/9/10.
with open("C:/rokey/python/Test/Day/pizza_file1.txt","w",encoding='UTF-8')as f:
    f.write("페퍼로니피자 3000\n")
    f.write("치즈피자 3200\n")
    f.write("콤비네이션피자 3500\n")

with open("C:/rokey/python/Test/Day/pizza_file1.txt","a",encoding='UTF-8')as f:
    f.write("불고기피자 3600\n")
    f.write("해산물피자 3800\n")

with open("C:/rokey/python/Test/Day/pizza_file1.txt","r",encoding='UTF-8')as f:
    lines=f.readlines()
    p_list=[]
    for line in lines:
        plist=line.split()
        pizza=plist[0]
        p_list.append(pizza)
        cost=plist[1]
    print(p_list)