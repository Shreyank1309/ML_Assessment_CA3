MACHINE LEARNING LAB ASSESSMENT
SARCASM DETECTION USING NEWS HEADLINES

INTRODUCTION:
The detection of sarcasm is a significant research problem in Natural Language Processing (NLP). Human communication often includes hidden meanings, emotions, and indirect expressions. While humans can easily understand sarcasm due to tone, context, and experience, machines find it difficult because they process plain text without context or tone.

For example, the sentence “Oh great, another exam tomorrow” appears positive, but actually expresses frustration — a sarcastic remark. This project aims to address such challenges using Machine Learning (ML) and Deep Learning (DL) models to automatically classify news headlines as sarcastic or non-sarcastic.

This lab project focuses on building and comparing two approaches:

Classical Machine Learning using TF-IDF and Logistic Regression

Deep Learning baseline using LSTM

The process involves data preprocessing, text vectorization, model training, evaluation, and comparison of performance between both methods.

PROJECT OBJECTIVE:
The main objectives of this project are:
• To load and analyze the News Headlines Sarcasm dataset.
• To preprocess and clean the text data.
• To convert the text into numerical features using TF-IDF.
• To train and evaluate classical machine learning algorithms.
• To implement a baseline LSTM deep learning model for comparison.
• To determine the most effective model for sarcasm detection.

DATASET DESCRIPTION:
The dataset used in this project is the “News Headlines Dataset for Sarcasm Detection.”
It contains thousands of real news headlines collected from websites such as TheOnion.com and HuffingtonPost.com.

Each data entry contains:
• headline – the text of the news headline
• is_sarcastic – binary label (1 = sarcastic, 0 = not sarcastic)

This dataset is popular because it contains diverse, short, and balanced text samples, making it ideal for sarcasm classification experiments.

NLP CHALLENGE IN SARCASM DETECTION:
Sarcasm detection is complex because it often lacks clear indicators and depends heavily on contextual understanding.
Unlike sentiment analysis, where positive or negative words define tone, sarcastic sentences often use positive words to convey negative meaning.

Example:
“Sure, I love standing in line for hours.”
Here, the word “love” is positive, but the intended meaning is frustration.

Hence, sarcasm detection requires intelligent feature representation and well-trained models to capture such hidden patterns.

METHODOLOGY:
The following workflow was followed for the implementation:

Step 1: Data Loading and Exploration
The dataset was loaded into Python using Pandas. Basic data inspection was done, including checking the number of samples, data distribution, and missing values.

Step 2: Data Preprocessing
Text was converted to lowercase, punctuation marks and unnecessary spaces were removed, and optional stopword removal was performed. These steps helped in cleaning and improving data quality.

Step 3: Feature Extraction using TF-IDF
Since ML models cannot directly understand text, TF-IDF (Term Frequency – Inverse Document Frequency) was used to convert text into numeric vectors. TF-IDF assigns higher weight to unique words and lower weight to common ones, making it an effective representation technique.

Step 4: Train-Test Split
The dataset was divided into training and testing subsets to evaluate performance on unseen data.

Step 5: Model Training
The Logistic Regression model was trained on the TF-IDF features. Logistic Regression is efficient, easy to implement, and suitable for binary text classification.

Step 6: Evaluation
Performance was measured using metrics such as Accuracy, Precision, Recall, F1 Score, and Classification Report.

Step 7: Deep Learning Baseline (LSTM)
An LSTM model was implemented using TensorFlow/Keras to compare performance. However, it achieved lower accuracy due to limited data and lack of parameter tuning.

RESULTS:
Two experiments were performed using TF-IDF + Logistic Regression models.

Results obtained:
• Accuracy for Dataset 1: 0.8297
• Accuracy for Dataset 2: 0.8398
• LSTM model achieved approximately 0.5232 accuracy (poor performance)

INTERPRETATION OF RESULTS:
The TF-IDF + Logistic Regression model achieved the highest accuracy of around 84%.
This shows that classical machine learning techniques can outperform deep learning for small or moderate text datasets.

Deep learning models like LSTM generally require larger datasets, proper hyperparameter tuning, and more computational resources to perform well.

Hence, for this task, Logistic Regression with TF-IDF representation proved to be the most effective approach.

CONCLUSION:
This project successfully demonstrates a complete Machine Learning pipeline — from data loading and preprocessing to model training and evaluation.

The major findings are:
• Simple and classical ML models can perform better than deep learning models for small NLP datasets.
• TF-IDF features capture textual information efficiently and improve accuracy.
• Proper preprocessing and model selection play a vital role in performance.

The best result obtained was approximately 84% accuracy using TF-IDF with Logistic Regression.
Future work can include using advanced NLP models such as BERT or transformer-based architectures to achieve higher sarcasm detection accuracy.
