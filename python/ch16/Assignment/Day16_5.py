def PN(expressions):
    stack = []
    oper = ["+","-","*","/"]
    for char in expressions: 
        if char not in oper: 
            stack.append(int(char)) 
        else:
            b = stack.pop()   
            a = stack.pop()
            if char == "+" :
                stack.append(a+b)
            elif char == "-" :
                stack.append(a-b)
            elif char == "*" :
                stack.append(a*b)
            elif char == "/" :
                stack.append(a/b)
    return stack[0]
            
print(PN("34+"))