# Email-Spam-Detection




# Overview 

This project implements a machine learning pipeline to classify emails as spam or ham(non-spam) using a Multinomial Naive Bayes statistical classifier with TF-IDF(Term Frequence-Inverse Document Frequency) features. The pipleline includes data preprocessing, feature extraction, model training, evaluation and visualization. The project was tested on a Mac M1 environment and includes error handling and logging. 


# Dataset:

The model uses the combined_data.csv dataset - The Kaggle Spam Email Data with two columns:
- label: Binary labels( 0 for ham, 1 for spam)
- text: Email content(subject+ body).
- The dataset is sampled to 80 000 rows but is recommended to adjust for memory constraints.


# Requirements:
- Python 3.1x
- Libraries: pandas, numpuy, scikit-learn, matplotlib, seaborn, joblib

- Install dependencies

 `bash "pip install -r requirements.txt"
 `

# Usage:
1) Prepare Dataset: Place combined_data.csv in the project directory.
2) Run the script: 

`bash "python classifier.py"`

3) Output:
- Console logs with metrics (e.g., Accuracy: 0.92, Precision: 0.91).
- Plots: confusion_matrix.png, roc_curve.png, feature_importance.png.
- Saved files: spam_model.pkl, tfidf_vectorizer.pkl.




# Results

Metrics: Typical performance on spam datasets:
- Accuracy: 0.85–0.95
- Precision: 0.80–0.95
- Recall: 0.70–0.90
- F1-score: 0.75–0.90
- ROC-AUC: 0.90–0.98

Visualizations:
- Confusion matrix shows true/false positives/negatives.
- ROC curve visualizes model discriminability.
- Feature importance highlights top spam words (e.g., “free,” “offer”).
- Logs: Detailed logging for debugging and documentation.