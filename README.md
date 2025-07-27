# SENTIMENT-ANALYSIS
* COMPANY: CODTECH IT SOLUTIONS
* NAME: AISHWARYA ZUNJAR MARATHE
* INTERN ID: CITS0D676
* DOMAIN: DATA ANALYTICS
* DURATION: 6 WEEKS
* MENTOR: NEELA SANTOSH
* # Sentiment Analysis on Product Reviews using NLP

## Project Overview

This project focuses on performing **sentiment analysis** on product reviews using **Natural Language Processing (NLP)** techniques. The goal is to classify customer reviews as **Positive**, **Neutral**, or **Negative** based on the content of their written feedback.

## Objective

- Analyze and clean raw text review data
- Convert text into numerical features using NLP techniques
- Train a machine learning model to classify sentiment
- Evaluate the model's performance and draw useful insights

## Dataset

- The dataset contains thousands of product reviews.
- Each review includes:
  - Product ID
  - Review Text
  - Rating Score (1 to 5)
  - Additional metadata (User ID, Summary, etc.)
- Used the **"Text"** column as input and the **"Score"** column to derive sentiment labels.

## Sentiment Labeling

- Reviews are labeled based on the rating (`Score`) as:
  - **Positive**: Score 4 or 5
  - **Neutral**: Score 3
  - **Negative**: Score 1 or 2

## Key Steps

### 1. Data Preprocessing

- Removed missing values
- Converted text to lowercase
- Removed punctuation, numbers, and stopwords
- Applied tokenization and basic cleaning

### 2. Feature Extraction

- Used **TF-IDF Vectorization** to convert text into numeric vectors
- Limited the number of features to reduce complexity

### 3. Model Training

- Used **K-Nearest Neighbors (KNN)** classifier
- Split the data into training and testing sets
- Trained the model on vectorized data

### 4. Evaluation

- Measured model performance using:
  - Accuracy score
  - Confusion matrix
  - Classification report (Precision, Recall, F1 Score)

## Tools and Libraries Used

- Python
- Pandas
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Matplotlib and Seaborn (for visualization)

## Insights and Results

- The model was able to classify sentiments with high accuracy
- TF-IDF helped in extracting important words influencing sentiment
- Most reviews were positive, indicating a class imbalance

## Future Improvements

- Use more advanced models like Logistic Regression, SVM, or deep learning (e.g., BERT)
- Improve text preprocessing using lemmatization and named entity removal
- Add a web interface for live sentiment prediction
