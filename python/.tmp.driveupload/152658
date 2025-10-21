# 컴파일 옵션 3

import re
p = re.compile(r'&[#](0[0-7]+|[0-9]+|x[0-9a-fA-F]+);')

p = re.compile(r"""
    &[#]
    (
    0[0-7]
    |[0-9] 
    |x[0-9a-fA-F]+                     
    )
    ;
    """, re.VERBOSE)