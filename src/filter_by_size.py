import pandas as pd

# find . -iname "*.jpg" -type f -exec identify -format '%i,%w,%h\n' '{}' \; > /tmp/res.csv
filepath = './res.csv'
df = pd.read_csv(filepath, header=None, names=['file', 'w', 'h'])
print(df.shape)
small = df[(df.w < 20) | (df.h < 20)]
print(small.shape)
small.file.to_csv('/tmp/todel.txt', index=False)
