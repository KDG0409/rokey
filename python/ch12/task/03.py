f=open("C:/rokey/python/ch12/task/계좌1.txt",'w',encoding="utf-8")

f.write("김삿갓 597-89-000089\n")
f.write("이수근 343-64-000064\n")
f.write("박혁거세 136-97-000097\n")
f.write("강호동 147-12-002093\n")
f.write("유재석 146-22-102093\n")

f.close()

f=open("C:/rokey/python/ch12/task/계좌1.txt",'r',encoding="utf-8")
# flist = f.read().split()
onerlist = []
numlist = []
tlist={}
# for num in range(len(flist)):
#     if num %2 == 1 :
#         numlist.append(flist[num])
#     if num %2 == 0 :
#         onerlist.append(flist[num])

# for i in range(len(onerlist)):
#     tlist[onerlist[i]] = numlist[i]

lines = f.readlines()
print(lines)
for line in lines :
    lineList = line.split()
    onerlist.append(lineList[0])
    numlist.append(lineList[1])
    tlist[lineList[0]] = lineList[1] #딕셔너리 활용
    
for key,val in tlist.items():
    print(f"{key} : {val}")

print(onerlist)
print(numlist)
print(tlist)
