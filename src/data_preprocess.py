

import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, encoding='utf-8')
    df.set_index('행정동명', inplace=True)
    # 문자열 숫자를 숫자로 변환
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)
    return df

