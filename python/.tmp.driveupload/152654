import re

text = ' My phone number is 123-456-7890 '

p = re.compile(r'\d+') # 숫자만 반복해서 매칭
num = p.findall(text)
print(num)

text2 = "Contact us at support@example.com or sales@example.org."

p2 = re.compile(r"\w+@\w+\.\w+")
# p2 = re.compile(r"[\w.-]+@[\w.-]+\.\w+")
emailAdd = p2.search(text2)
print(emailAdd)

phone = "123-456-7890"

p3 = re.compile(r"\d{3}-\d{3}-\d{4}")
phoneNum = p3.match(phone)
if phoneNum != None:
    print("적절함")
else:
    print("부적절함")
