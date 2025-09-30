import re
p = re.compile(r"\w+@\w+.\w+")

text = "이메일 목록: test@example.com, hello@world.net, user123@domain.org"
m = p.findall(text)
print(m)