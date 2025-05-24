import pandas as pd
import numpy as np

#Load dataset

df = pd.read_csv('combined_data.csv', encoding="latin-1", header=None)

df.columns=['label', 'text']
df = df[['label', 'text']]

df = df.sample(n=5000, random_state=42)
df.describe(include="all")
