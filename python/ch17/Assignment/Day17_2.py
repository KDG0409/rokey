def myMap(n,m,data):
    ans=[]
    for i in range(n):
        row=[]
        for j in range(m):
            row.append(data.pop(0))
        ans.append(row)
    return ans

data = [0,1,1,1,1,1,0,1,0,0,0,1,0,1,0,1,0,1,
        0,1,0,1,0,0,0,0,0,1,1,0,1,1,1,1,1,0]
ans = myMap(6,6,data)
print(ans)