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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import logging 


#Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s- %(levelname)s - %(message)s')

#Download NLTK (run once)
try:
    nltk.download('stopwords')
except Exception as e:
    logging.error(f"Failed to download NLTK: {e}")
    raise


def preprocess_text(text):
    #Preprocess text by cleaning, lowercasing, removing stop words and implementing stemming.
    try:
        if not isinstance(text, str):
            logging.warning("No string input detected, returning empty string")
            return ""
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
    except Exception as e:
        logging.error(f"Error in preprocess_text: {e}")
        return ""

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    #plot confusion matrix using seaborn heat map

    c_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Ham (0)', 'Spam (1)'], yticklabels=['Ham (0)', 'Spam (1)'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()


def plot_roc_curve(y_true, y_scores, title="ROC CURVE"):
    #Plot ROC Curve and compute the Area Under Curve

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC CURVE ( AUC = {auc:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.close()

def plot_feature_importance(model, vectorizer, top_n = 30):
    #plot top spam and ham words based on model coefficients
    feature_names = vectorizer.get_feature_names_out()
    #MultinomialNB coefficients for class spam (class 1)
    log_probs = model.feature_log_prob_[0]
    top_spam_idx = np.argsort(log_probs)[-top_n:]
    top_spam_words = [feature_names[i] for i in top_spam_idx]
    top_spam_scores = [log_probs[i] for i in top_spam_idx]
    #plot
    plt.figure(figsize=(8,5))
    plt.barh(top_spam_words, top_spam_scores, color='red')
    plt.title(f'Top {top_n} Words Indicative of Spam')
    plt.xlabel('Coefficient Value')
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    #Main function to run spam pipeline
    try:
        #Load dataset
        df = pd.read_csv('combined_data.csv', encoding="latin-1", header=None)
        df.columns=['label', 'text']
        logging.info(f"Loaded dataset with shape: {df.shape}")
    except FileNotFoundError:
        logging.error("Dataset file 'combined_data.csv' not found")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise
    #Convert labels to boolean values
    df['label'] = df['label'].map({'0': 0, '1': 1})
    logging.info(f"Number of NaN labels: {df['label'].isna().sum()}")
    logging.info(f"Number of NaN texts: {df['text'].isna().sum()}")
    #Handle cases for Nan
    df = df.dropna(subset=['label', 'text'])
    df.describe(include="all")

    #sample dataset
    sample_size = 80000
    if df.shape[0] < sample_size:
        logging.warning(f"Dataset has {df.shape[0]} rows, less than requested {sample_size}")
        sample_size = df.shape[0]
    df = df.sample(n=sample_size, random_state=42)
    #logging.info(f"Class distribution:\n {df['label'].value_counts()}")

    #apply preprocess text
    df['cleaned_text'] = df['text'].apply(preprocess_text) 
    df['cleaned_text'].describe(include='all')
    
    #Feature Extraction  with TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    #    Fit numerical features 
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    logging.info(f"Feature matrix shape: {X.shape}")
    #Split data into training and testing sets and stratified split 
    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2, random_state=42)
    logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    #Train the model 
    model = MultinomialNB()
    model.fit(X_train,y_train)
    logging.info("Model training completed")
    
    #Evaluate the model 
    y_pred = model.predict(X_test)
    #probabilities for ROC
    y_scores = model.predict_proba(X_test)[:,1]
    #Proportion of correct predictions
    accuracy = accuracy_score(y_test, y_pred)
    #accuracy of positive predictions 
    precision = precision_score(y_test, y_pred)
    #Ability for detection 
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_scores)
    
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1:.4f}")

    #Visualizations

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_scores)
    plot_feature_importance(model, vectorizer)
 
    #Save modek and vectorizer for deployment

    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    logging.info("Model saved")


if __name__=="__main__":
    main()

## %%
