import re

p = re.compile('[a-z]+') #알파벳이 반복될 때 까지 매칭 = 단어 매칭

# match메서드 

m = p.match("python")
print(m)

m = p.match("3python")
print(m)

# search메서드 

m = p.search("python")
print(m)

m = p.search("3python")
print(m)

# findall 메서드

m = p.findall("python 3python")
print(m)

m = p.findall("3python")
print(m)

# finditer 메서드

m = p.finditer("python 3python")
print(next(m))
print(next(m))
for r in m:
    print(r)
for r in m:
    print(r.group())

m = p.finditer("3python")
for r in m:
    print(r)