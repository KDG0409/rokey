# import re

# pattern = r'\w+'
# text = "Hello, World!"
# print(re.findall(pattern, text))

import re
pattern = r'(ab)+'
text = "ababab"
match = re.match(pattern, text)
print(match.group())
