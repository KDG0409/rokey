def valsum(dic):
    valuea = dic.values()
    sum = 0
    for val in valuea :
        int(val)
        sum += val
    return sum

dicta = { 'a': 10, 'b': 20, 'c': 30 }
ans = valsum(dicta)
print(ans)

