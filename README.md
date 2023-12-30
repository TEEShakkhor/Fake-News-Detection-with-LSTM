
# Fake news detection with Naive Bayes, Logistic regression, Decision tree and LSTM

---

## Overview

The **Fake News Detection project** is a machine learning endeavor aimed at identifying and classifying news articles as either real or fake. The project utilizes various algorithms, including **Naive Bayes, Logistic Regression, Decision Tree, and LSTM Neural Network**, to achieve accurate and reliable results. By leveraging a diverse set of models, the project provides a comprehensive approach to addressing the challenge of misinformation in the digital age.

----

## Project Structure

The project is organized into distinct components, each serving a crucial role in the overall workflow:

1. **Data Preprocessing:**
   - Removal of unnecessary columns and null values.
   - Text cleaning, including the removal of special characters, punctuation, and stopwords.
   - Visualization of frequent words through WordCloud.

2. **Model Implementation:**
   - Implementation of Naive Bayes, Logistic Regression, Decision Tree, and LSTM Neural Network models.
   - Training and evaluation of each model using performance metrics.
   - Visualization of confusion matrices for result interpretation.

3. **Model Testing:**
   - Testing the trained models on unseen input for real-time detection.
   - Providing a script for users to input their own news text and receive predictions.

---

## Setup Instructions

### Prerequisites

- **Python Version:** 3.10.0
- **CUDA Version:** 11.2
- **cudNN Library Version:** 8.1

### Dataset

- Download the [Fake News Dataset](https://git.io/J0fjL) and place it in the project directory.

### Virtual Environment Setup

1. Set up a virtual environment:

    ```bash
    python -m venv [name]
    ```

2. Activate the virtual environment:

    - On Windows:

    ```bash
    [name]\Scripts\activate
    ```

    - On Unix or MacOS:

    ```bash
    source [name]/bin/activate
    ```

### Package Installation

Install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
---

## Documentation

### 1. Data Preprocessing

The project begins with a thorough data preprocessing phase, cleaning and preparing the dataset for model training. Special attention is given to the removal of irrelevant information, handling missing values, and cleaning the text data.

### 2. Model Implementation

The project employs four distinct models for fake news detection: Naive Bayes, Logistic Regression, Decision Tree, and LSTM Neural Network. Each model is implemented, trained on the dataset, and evaluated for performance.

#### 2.1 Multinomial Naive Bayes Model

```python
model_nb = MultinomialNB()
model_nb.fit(x_train_tfidf, y_train)
y_pred_nb = model_nb.predict(x_test_tfidf)
#... (Performance evaluation and confusion matrix visualization)
```

#### 2.2 Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(x_train_tfidf, y_train)
y_pred_lr = model_lr.predict(x_test_tfidf)
#... (Performance evaluation and confusion matrix visualization)
```

#### 2.3 Decision Tree Model

```python
from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier()
model_dt.fit(x_train_tfidf, y_train)
y_pred_dt = model_dt.predict(x_test_tfidf)
# ... (Performance evaluation and confusion matrix visualization)
```

#### 2.4 LSTM Neural Network

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# ... (Tokenization and padding)
# ... (Splitting data into test and train set for LSTM)
# ... (Creating and training the LSTM model)
# ... (Model performance and accuracy evaluation)
```

### 3. Model Testing

Users can test the models on their own input for real-time fake news detection. The provided scripts guide users through the process, ensuring seamless and accessible testing.

---

## Conclusion

The Fake News Detection project offers a robust and diverse set of models to effectively combat the spread of misinformation. Users are encouraged to explore the provided code for a deeper understanding and contribute to advancing fake news detection algorithms.

---

*Note: For detailed code implementation and analysis, refer to the corresponding notebooks and scripts within the repository.*
