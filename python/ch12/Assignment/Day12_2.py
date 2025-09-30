with open("C:/rokey/python/ch12/Assignment/pizza_file1.txt",'w', encoding='utf-8') as f:
    f.write("페퍼로니피자 3000\n")
    f.write("치즈피자 3200\n")
    f.write("콤비네이션피자 3500\n")
    f.write("불고기피자 3600\n")
    f.write("해산물피자 3800\n")

# with open("C:/rokey/python/ch12/Assignment/pizza_file1.txt",'r', encoding='utf-8') as f:
#     lines = f.read().split()
#     ptype = []
#     for i in range(len(lines)):
#         if i %2 == 0 :
#             ptype.append(lines[i])
#     print(ptype)

with open("C:/rokey/python/ch12/Assignment/pizza_file1.txt",'r', encoding='utf-8') as f:
    lines = f.readlines()
    ptype = []
    clist = []
    for line in lines:
        lineList = line.split()
        ptype.append(lineList[0])
        clist.append(lineList[1])
    print(ptype)
    print(clist)