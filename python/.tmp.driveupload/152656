# 컴파일 옵션 2

import re
p = re.compile("^python\s\w+") #python 이후 화이트 스페이스 이후 단어(w+)

data = """python one
life is too short
python two
you need python
python three"""

print(p.findall(data)) #첫 번째 줄만 매치 후 종료 (^,$가 전체 중을 의미)

p2 = re.compile("^python\s\w+", re.MULTILINE) # re.MULTILINE 또는 re.M
data = """python one
life is too short
python two
you need python
python three"""
print(p2.findall(data)) # ^,$가 라인별을 의미)