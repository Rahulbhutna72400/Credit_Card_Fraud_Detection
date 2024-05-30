# Credit Card Fraud Detection

This project focuses on building a machine learning model to detect fraudulent credit card transactions. The dataset used contains information about credit card transactions, including whether each transaction is fraudulent or legitimate. Various machine learning algorithms, including Logistic Regression, Decision Trees, and Random Forests, are employed to classify transactions.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Description

Credit card fraud detection is a critical task in the financial industry. This project aims to create a robust model that can accurately classify transactions as fraudulent or legitimate. The dataset includes various features such as transaction amount, transaction time, and cardholder information.

### Key Features:
1. **Data Preprocessing**:
   - Handling missing values
   - Feature extraction from datetime fields
   - One-hot encoding of categorical variables
  Dataset - https://www.kaggle.com/datasets/kartik2112/fraud-detection

2. **Model Training**:
   - Training multiple models (Logistic Regression, Decision Tree, Random Forest)
   - Hyperparameter tuning for the Random Forest model

3. **Model Evaluation**:
   - Confusion matrix
   - ROC curve and AUC score
   - Feature importance analysis for the Random Forest model

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preprocessing**:
   The data preprocessing script handles missing values, extracts useful features from datetime fields, and applies one-hot encoding to categorical variables.

   ```python
   import pandas as pd
   from preprocess import preprocess_data

   train_data = pd.read_csv('data/fraudTrain.csv')
   test_data = pd.read_csv('data/fraudTest.csv')

   X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)
