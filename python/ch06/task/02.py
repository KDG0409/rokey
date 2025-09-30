# 양수 음수 구분

def fplusminus(arg):
    if arg > 0 :
        return "plus"
    elif arg < 0 :
        return "minus"
    else :
        return "zero"
    
stra = fplusminus(0)
print(stra)