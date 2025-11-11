MACHINE LEARNING LAB ASSESSMENT
SARCASM DETECTION USING NEWS HEADLINES
INTRODUCTION

The idea of detecting sarcasm is a significant research issue in Natural Language Processing since human communication is often characterized by hidden opinions, feelings, and indirect meanings. Humans can easily identify sarcasm using context, tone, and experience, but machines find it difficult because they process plain text without any tonal or emotional context.

For instance, the sentence “Oh great, another exam tomorrow” appears positive but actually conveys frustration — a sarcastic remark. This project aims to address this challenge using Machine Learning and Deep Learning models to automatically classify sarcasm in text.

The project involves constructing models that categorize news headlines as sarcastic or non-sarcastic. It compares classical ML models (TF-IDF + Logistic Regression) with a Deep Learning baseline (LSTM). The workflow includes data preprocessing, feature extraction, training, evaluation, and model comparison.

PROJECT OBJECTIVE

The main objective is to design and test models that can identify sarcasm in news headlines.
The specific goals include:

Loading and understanding the sarcasm dataset.

Performing thorough text preprocessing.

Converting text into numerical feature representations.

Implementing classical machine learning algorithms and evaluating their performance.

Applying a deep learning model for comparison.

Determining which model performs best for sarcasm detection.

DATASET DESCRIPTION

The dataset used in this research is the News Headlines Dataset for Sarcasm Detection.
It contains thousands of news headlines collected from websites such as TheOnion.com and HuffingtonPost.com.

Each entry consists of:

headline → The news headline (string)

is_sarcastic → Binary label (1 = sarcastic, 0 = not sarcastic)

This dataset is widely used because it contains short, diverse, and balanced text samples, making it ideal for sarcasm detection research.

NLP CHALLENGE IN SARCASM DETECTION

Sarcasm is difficult to detect due to lack of background information, hidden motives, and indirect meaning.
In sentiment analysis, simple word frequency features may suffice, but sarcasm differs because sarcastic sentences often use positive words to express negative sentiments.

Example: “Sure, I love standing in line for hours.”
Although the word “love” is positive, the intended meaning is frustration.
Hence, efficient feature representation and carefully trained models are required to capture these hidden patterns.

METHODOLOGY

The following steps were followed to develop and evaluate the sarcasm detection models:

Step 1: Data Loading and Exploration

The dataset was loaded using Pandas in Python.

Basic exploration included total record count, class distribution, and sample inspection.

Step 2: Data Preprocessing

Converted text to lowercase.

Removed punctuation and extra spaces.

Optionally removed stopwords.
These steps helped clean the data and improve model performance.

Step 3: Feature Extraction using TF-IDF
Since ML models cannot interpret text directly, TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert words into numerical vectors.
TF-IDF assigns higher weight to rare but meaningful words, improving feature quality.

Step 4: Train–Test Split
The dataset was split into training and testing subsets to evaluate performance on unseen data.

Step 5: Model Training (Logistic Regression)
A Logistic Regression model was trained using TF-IDF features.
This algorithm is efficient, fast, and highly suitable for binary text classification.

Step 6: Evaluation Metrics
Performance was measured using:

Accuracy

Precision

Recall

F1 Score

Classification Report

Step 7: Deep Learning Baseline (LSTM)
A simple LSTM model was implemented using TensorFlow and Keras.
However, this model achieved low accuracy because deep learning requires large datasets, multiple epochs, and extensive hyperparameter tuning.

RESULTS

Two experiments were conducted using the TF-IDF + Logistic Regression approach.

Model	Dataset	Accuracy
Logistic Regression (TF-IDF)	Dataset-1	0.8297
Logistic Regression (TF-IDF)	Dataset-2	0.8398
LSTM (Deep Learning Baseline)	-	0.5232
INTERPRETATION OF RESULTS

The TF-IDF + Logistic Regression model gave the best results, achieving an accuracy of around 84%.
This proves that classical ML models can outperform deep learning models for short text classification problems like sarcasm detection.

Deep learning (LSTM) requires large training data, careful tuning, and more computational resources to perform better. In contrast, TF-IDF combined with Logistic Regression provided high accuracy with simplicity and efficiency.

CONCLUSION

This project successfully demonstrates a complete Machine Learning pipeline — from data collection and preprocessing to model training and evaluation.

Key takeaways include:

Classical ML models (like Logistic Regression) can outperform Deep Learning in smaller datasets.

TF-IDF feature representation effectively captures textual sarcasm patterns.

Deep learning models need more training data, hyperparameter tuning, and computational power.

The Logistic Regression model achieved approximately 84% accuracy, proving to be the most effective method.

Future improvements could include using transformer-based models (like BERT) or contextual embeddings to further enhance sarcasm detection accuracy.
