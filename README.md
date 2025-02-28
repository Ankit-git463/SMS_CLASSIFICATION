# SMS Classification Model

A machine learning project to classify SMS messages as spam or legitimate (ham).

## Table of Contents

* Overview
* Methodology
  * 1. Data Cleaning
  * 2. Exploratory Data Analysis
  * 3. Text Preprocessing
  * 4. Model Building
  * 5. Evaluation
* Results

## Overview

This project aims to build a machine learning model that can accurately classify SMS messages as spam or ham (legitimate). 


## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/sms-classification.git
cd sms-classification
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Methodology

### 1. Data Cleaning

* Loading the SMS dataset
* Handling missing values
* Removing duplicates
* Basic text cleaning

### 2. Exploratory Data Analysis

Using NLTK and Seaborn to analyze:

* Distribution of spam vs ham messages
* Message length comparison
* Word frequency analysis
* Common patterns in spam messages
* Correlation heatmaps of features
* Pairplots for numerical features
* Word clouds for spam and ham messages

### 3. Text Preprocessing

Several techniques were applied to prepare the text data:

* Converting all text to lowercase
* Tokenization (breaking text into individual words/tokens)
* Removing special characters and numbers
* Removing stop words and punctuation
* Stemming (reducing words to their root form)
* Creating a bag-of-words model or TF-IDF vectors

### 4. Model Building

Several classification algorithms were tested:

* Naive Bayes (MultinomialNB)
* Support Vector Machine (SVM)
* Random Forest
* Logistic Regression
* Decision Trees
* AdaBoost
* ExtraTrees Classifier
* Gradient Boosting Classifier

### 5. Evaluation

Models were evaluated using:

* Accuracy, Precision, Recall, F1-score
* Confusion Matrix
* Cross-validation to ensure reliability

### 6. Improvement

Techniques used to improve model performance:

* Hyperparameter tuning
* Feature engineering
* Ensemble methods

## Results

Our best performing model achieved is :

* Accuracy: 97.0019%
* Precision: 100%

The model performs exceptionally well in identifying spam messages, with a particularly high precision rate, minimizing false positives.
