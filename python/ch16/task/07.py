# 괄호 짝 검사
# 빈 스택 생성->표현식 내 문자를 가져와 문자에 ""가 있으면 스택에 push
# ""괄호가 있다면 스택에 pop->스택이 비었다면 False, pop한 내용이 일치하지 않으면 False
# 앞에서 짝이 틀리지 않았다면 True
# 시험x 난이도 높음
# 코드 해석 

def check_brackets(expressions):
    stack = []
    for char in expressions: #문자열 순회
        if char in "({[": #문자열에 열림 괄호가 나오면/True이면 추가
            stack.append(char) #스택에 추가
        elif char in ")}]": #문자열에 닫힘 괄호가 나오면/True이면 비교
            if not stack: #스택이 빈 경우 False
                return False
            top = stack.pop() # 최근 열림괄호 내용 추출,스택에서 제거
            if char == ")" and top != "(": # char:닫힘 괄호 top:열림 괄호
                return False
            elif char == "}" and top != "{":
                return False            
            elif char == "]" and top != "[":
                return False
    return len(stack) == 0 # 스택이 비어있으면 True
            
print(check_brackets("(a+b)"))
print(check_brackets("([a+b])"))