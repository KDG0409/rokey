def fmax(ca):
    max = ca[0]
    for i in range(1,len(ca)):
        if max < ca[i] :
            max = ca[i]
    return max

a=[2,3,67,543,3233]
ans = fmax(a)
print(ans)
