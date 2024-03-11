import pandas as pd
from typing import List

INPUT_FILES: List[str] = [
    "./volumes/data/test_matched-00000-of-00001.parquet",
    "./volumes/data/test_mismatched-00000-of-00001.parquet",
    "./volumes/data/train-00000-of-00001.parquet",
    "./volumes/data/validation_matched-00000-of-00001.parquet",
    "./volumes/data/validation_mismatched-00000-of-00001.parquet"
]
OUTPUT_FILE: str = "./volumes/data/glue_mnli.csv"


def concat_texts(row) -> str:
    return row['premise'] + ";" + row['hypothesis']


dfs = [pd.read_parquet(file) for file in INPUT_FILES]
df = pd.concat(dfs)
df['text'] = df.apply(concat_texts, axis=1)
df = df.drop(columns=['premise', 'hypothesis'])
df = df[df.label != -1]
df = df.sample(frac=1)
df = df.head(n=2000)

df.to_csv(OUTPUT_FILE, index="idx")
