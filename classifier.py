# %%
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Run once
nltk.download('stopwords')


#Load dataset
df = pd.read_csv('combined_data.csv', encoding="latin-1", header=None)

df.columns=['label', 'text']
df = df[['label', 'text']]

df = df.sample(n=5000, random_state=42)
df.describe(include="all")


def preprocess(text):
    #Remove all special characters and noise keep only letters and space
    text = re.sub(r'[^A-Za-z\s]', '', text)
    #convert all text to lowercase to ensure consistency
    text = text.lower()
    words = text.split()
    #remove all stopwords: those that have little or no sentiment 
    stop_words = set(stopwords.words('english'))
    words = [words for word in words if word not in stop_words]
    #apply stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)