# Employee Retention Classification - A Classification Challenge with ML and Deep Learning

## Introduction
This repository contains code for a classification challenge in employee retention using Machine Learning and Deep Learning techniques. The goal is to predict whether an employee will leave or stay with a company based on various features.

### Table of Contents
1. [General Information](#general-information)
2. [About the Dataset](#about-the-dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Train-Test Split](#train-test-split)
5. [Exploring the Target Variable](#exploring-the-target-variable)
6. [Handling Class Imbalance](#handling-class-imbalance)
7. [XGBoost and Random Forest](#xgboost-and-random-forest)
8. [Neural Network](#neural-network)
9. [Testing and Evaluating Models](#testing-and-evaluating-models)
10. [Conclusion](#conclusion)

---

### General Information

#### Importing General Libraries
The code starts by importing necessary Python libraries for data manipulation, visualization, preprocessing, model building, evaluation, and deep learning using TensorFlow and Keras.

#### About Our Classification Dataset
This dataset contains information about employees in a company, including their educational backgrounds, work history, demographics, and employment-related factors. It has been anonymized to protect privacy while still providing valuable insights into the workforce. The source of the dataset is Kaggle.

- Target Column: LeaveorNot (a binary classification target).
- Numeric Columns: Age, JoiningYear, ExperienceinCurrentDomain, and Salary.
- Categorical Columns: PaymentTier, Education, Gender, City, and EverBenched.

#### Loading Data and Preprocessing
The code loads the dataset from a URL and provides an overview of the data's structure and summary statistics. Data preprocessing steps include encoding categorical features using one-hot encoding.

#### Train-Test Split
The data is split into training and testing sets to train and evaluate machine learning models.

---

### Exploring the Target Variable
The code provides visualizations and statistics to explore the distribution of the target variable, which indicates class imbalance.

---

### Handling Class Imbalance
Class imbalance is addressed using the ADASYN oversampling algorithm, which generates synthetic samples for the minority class.

---

### XGBoost and Random Forest
Two machine learning models, XGBoost and Random Forest, are applied to the data. Hyperparameter tuning is performed using GridSearchCV to find the best hyperparameters for each model.

---

### Neural Network
A neural network model is defined using Keras. The model architecture consists of input, hidden, and output layers. The model is compiled using the Adam optimizer and binary cross-entropy loss.

---

### Testing and Evaluating Models
The code evaluates the performance of the models on the test set, including Random Forest and the Neural Network. Classification reports are generated to assess model performance in terms of precision, recall, and F1-score.

---

### Conclusion
The conclusion summarizes the model evaluations and suggests that XGBoost performs well, especially in terms of recall, making it a suitable choice for predicting employee retention.

---

**Note:** This is a brief overview of the code. Detailed explanations, variable definitions, and data analysis can be found in the code provided.
