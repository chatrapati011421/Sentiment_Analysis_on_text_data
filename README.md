# Sentiment_Analysis_on_text_data

# Restaurant Review Sentiment Analyzer
This project is a simple machine learning-based sentiment analysis tool. It reads restaurant reviews and predicts whether they are positive or negative.

# Step-by-Step Instructions

# 1. Requirements

Make sure Python is installed on your system. You’ll also need to install a few Python libraries: pandas, numpy, nltk, scikit-learn, matplotlib, and seaborn.
These libraries are used for data handling, natural language processing, machine learning, and visualizing the results.

# 2. Project Files
**This project includes:**

"The dataset used in this project was sourced from Kaggle, a widely trusted platform known for high-quality datasets and machine learning competitions."

The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/ehabashraf/restaurant-reviewstsv/data), a trusted platform for high-quality datasets and machine learning projects.

The dataset must be placed in the same folder as the script before running it.

# 3. How the Project Works
  **The script performs the following tasks:**

  1.Loads the dataset containing restaurant reviews and labels

  2.Cleans and preprocesses the review text (removing punctuation, lowercasing, removing stopwords, and stemming)

  3.Converts the cleaned text into numerical features using TF-IDF vectorization

  4.Splits the data into training and testing sets

  5.Trains a Naive Bayes classifier on the training data

  6.Evaluates the model using a classification report and confusion matrix

  7.Predicts the sentiment of new example reviews
  
# 4. Output Visualization
**The script displays:**

  1.A classification report showing precision, recall, and F1-score

  2.A confusion matrix plot using Seaborn for better visual understanding

# 5. Technologies Used
**This project uses the following tools and libraries:**

  1.Python

  2.Pandas and NumPy for data manipulation

  3.NLTK for text preprocessing

  4.Scikit-learn for machine learning and evaluation

  5.Matplotlib and Seaborn for plotting

