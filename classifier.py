# %%
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Run once
nltk.download('stopwords')


#Load dataset
df = pd.read_csv('combined_data.csv', encoding="latin-1", header=None)

df.columns=['label', 'text']
df = df[['label', 'text']]
df['label'] = df['label'].map({'0': 0, '1': 1})

df = df.sample(n=80000, random_state=42)
print("Number of NaN labels:", df['label'].isna().sum())
df = df.dropna(subset=['label'])
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
#Train the model 
model = MultinomialNB()
model.fit(X_train,y_train)

#Evaluate the model 
y_pred = model.predict(X_test)
#Proportion of correct predictions
accuracy = accuracy_score(y_test, y_pred)
#accuracy of positive predictions 
precision = precision_score(y_test, y_pred)
#Ability for detection 
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")



#test 
#print(preprocess_text("Hello@ I am sending yoU a Ema@il"))

## %%
