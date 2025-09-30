# 문자열 길이 계산 함수 정의

def string_length(stb):
    count = 0
    for char in stb :
        count += 1
    return count

stb = input( "문자열을 입력하세요. : ")
lenb = string_length(stb)
print(lenb)