import pandas as pd
# 前处理
df = pd.read_csv('./data/bart_10_train.csv')
print(len(df))

df['q_length'] = df['query'].apply(lambda x: len(str(x)))
df['d_length'] = df['doc'].apply(lambda x: len(str(x)))

df = df[df['q_length']>=2]
print(len(df))
df['doc'] = df['doc'].apply(lambda x: x.replace('"', ''))  # 去掉 “
df['doc'] = df['doc'].apply(lambda x: x.replace('“', ''))
df = df[df['d_length']>=2]
print(len(df))
df[['query', 'doc']].to_csv('./data/bart_10_train.csv', index=False)


