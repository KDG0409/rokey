#최대값 찾기(함수)

def fmax(fu) :
    max = fu[0]
    for sb in fu :
        if max < sb :
            max = sb
    return max

su = [5,4,7,10,6]
max=fmax(su)
print(max)