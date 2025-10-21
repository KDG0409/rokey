# 컴파일 옵션 1
import re

p = re.compile('a.b')
m = p.match('a\nb')
print(m)

p = re.compile('a.b',re.S) #re.DOTALL 또는 re.S / .에 줄바꿈도 포함
m = p.match('a\nb')
print(m)

p = re.compile('[a-z]+', re.I) #re.IGNORECASE 또는 re.I / 대소문자 무시
p.match('python')
p.match('Python')
p.match('PYTHON')