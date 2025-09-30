import re

data = "user@example.com"
p = re.compile(r'\w+@\w+.\w+')
m = p.match(data)

if m:
    print("올바른 형식입니다.")
else :
    print("잘못된 형식입니다.")