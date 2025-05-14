import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')

# Load dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", sep='\t', quoting=3)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    stemmer = PorterStemmer()
    filtered_words = [stemmer.stem(w) for w in words if w not in stopwords.words("english")]
    return " ".join(filtered_words)

# Apply preprocessing
dataset['Cleaned_Review'] = dataset['Review'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
features = vectorizer.fit_transform(dataset['Cleaned_Review']).toarray()
labels = dataset['Liked'].values

# Split the data using custom variable names
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Train the Naive Bayes model
classifier = MultinomialNB()
classifier.fit(train_features, train_labels)

# Predict using the model
predicted_labels = classifier.predict(test_features)

# Evaluate the model
print(classification_report(test_labels, predicted_labels))

# Generate and plot confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", linewidths=0.5, linecolor='black',
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.show()

# Sentiment prediction function
def analyze_sentiment(new_text):
    clean_text = preprocess_text(new_text)
    vec_text = vectorizer.transform([clean_text]).toarray()
    return "Positive" if classifier.predict(vec_text)[0] == 1 else "Negative"

# Example usage
examples = [
    "I loved the meal, everything was perfect!",
    "Worst service and terrible food.",
    "Tried for 1st time was very good and delicious."
]

for review in examples:
    print(f"Review: {review}\nPrediction: {analyze_sentiment(review)}\n")