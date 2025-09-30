# 최소값 찾기

def fmin(fu):
    min = fu[0] 
    for sb in fu :
        if sb < min :
            min = sb
    return min

su = [5,4,7,10,6]
mina = fmin(su)
print(mina)