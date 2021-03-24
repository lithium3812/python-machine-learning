import pyprind
import pandas as pd
import os
from numpy.random import default_rng


basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for sd in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, sd, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                text = infile.read()
            df = df.append([[text, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

rng = default_rng()
df = df.reindex(rng.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

df = pd.read_csv('movie_data.csv', encoding='utf-8')
print("Top3 rows:\n", df.head(3))
print("Shape of the dataframe:", df.shape)