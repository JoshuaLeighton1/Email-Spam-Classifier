# %%
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#Run once
nltk.download('stopwords')


#Load dataset
df = pd.read_csv('combined_data.csv', encoding="latin-1", header=None)

df.columns=['label', 'text']
df = df[['label', 'text']]

df = df.sample(n=5000, random_state=42)
df.describe(include="all")


def preprocess_text(text):
    #Remove all special characters and noise keep only letters and space
    text = re.sub(r'[^A-Za-z\s]', '', text, flags=re.MULTILINE)
    #convert all text to lowercase to ensure consistency
    text = text.lower()
    words = text.split()
    #remove all stopwords: those that have little or no sentiment 
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    #apply stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

#Apply preprocesing

df['cleaned_text'] = df['text'].apply(preprocess_text) 

df['cleaned_text'].describe(include='all')

#Feature Extraction  with TF-IDF

vectorizer = TfidfVectorizer(max_features=5000)
#Fit numerical features 
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

#Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)





#test 
#print(preprocess_text("Hello@ I am sending yoU a Ema@il"))
