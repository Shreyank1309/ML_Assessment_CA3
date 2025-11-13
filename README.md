TITLE: MACHINE LEARNING LAB ASSESSMENT – SARCASM DETECTION USING NEWS HEADLINES
INTRODUCTION:
The idea of detecting sarcasm is a significant research issue in Natural Language Processing since human communication is frequently characterized by the hidden opinions, feelings, and indirect meaning. Everyday human beings have little difficulty in deciphering sarcasm due to our experience, context, tone, expressions and social knowledge. However, with machines, sarcasm is very hard due to the fact that the model perceives plain text with no context or tonal information. Let us take the example of the sentence Oh great, another exam tomorrow; it appears positive but then the meaning is negative and sarcastic. The proposed project attempts to address such an issue through machine learning algorithms and deep learning models to classify sarcasm automatically.
This lab evaluation consists of constructing models with the purpose of categorizing news headlines as sarcastic or non-sarcastic. We discuss classical machine learning with TF IDF and Logistic Regression and we develop also a baseline LSTM on deep learning. The assignment is to process the data, turn the text into numeric characteristics, model-train, and evaluate the performance and compare the performance of the two methods.
PROJECT OBJECTIVE:
This project aims at creating and testing machine learning models that identify sarcasm in news headlines. The key goals are:
• Load and learn sarcasm news headlines data.
• Do adequate text data preprocessing.
• Text to feature representation.
• Implement classical ML algorithms and- perform evaluation.
• Apply a deep learning baseline and compare the results.
• Determine what model is the most effective in this kind of problem.
DATASET DESCRIPTION:
The dataset that will be utilized in this research is the renowned popular dataset of News Headlines Dataset to Sarcasm Detection. It has thousands of revelational news headlines which are taken over dozens of news web sites such as theonion.com and huffingtonpost.com. There is a binary label and a headline string in each entry. In case a headline is sarcastic or in the style of fake humor, the label is 1. In case the headline is solemn and real news, the mark is 0. This dataset is also popular in the research community due to the high level of variety, short texts, and equal amount of sarcastic humor.

NLP CHALLENGE in SARCASM DETECTION:
Sarcasm is very difficult to detect because there is no background information, hidden motive and indirect meaning. Simple word frequency feature is typically sufficient in regular sentiment analysis tasks like positive/negative review. However, sarcasm is different since sarcastic sentences are constructed with positive words and the meaning of the sentence is negative. Using the example, the sentence Sure, I love standing in line, hours is made up of positive words such as love, however, it means frustration in reality. Thus we require feature representation and machine learning models that are well-constructed to ensure that we capture the hidden patterns.
METHODOLOGY:
The next workflow was carried out:
First, Data Loading and Exploration: This step will be carried out on the initial day of the project.
Step 1: Data Loading and Exploration: This task will be conducted during the first day of the project.
Pandas is used to load the dataset into Python. Simple examination like additional examination of total rows, quantity of sarcastic/non sarcastic samples, analysis of missing values, and examination of sample headlines was performed.
Step 2: Data Preprocessing
There are steps of text cleaning, such as to lowercase, delete punctuation marks, delete superfluous space and optional stopword deletion. These procedures enhance the stability of the model since raw texts involve noise.
Step 3: TF-IDF Extraction of features.
Because machine learning algorithms cannot read a text, the text can be transformed into numeric vectors with TF-IDF (Term Frequency -Inverse Document Frequency) vectorization. The TF-IDF assigns more weight to meaningful words and little to common words such as the and and. This step of feature extraction is highly important as performance is very much dependent on the quality of features.
Step 4: Train-Test Split
The data set is classified as training and test set in such a way that the performance of the model is appropriately tested on a new set of data.
Step 5: Model Training
TF-IDF vectors called vectors were trained in a Logistic Regression model. Among the most efficient algorithms to be used in text based problems involving binary classification, there is the Logistic Regression. It is easy, quick and highly efficient.
Step 6: Evaluation
Some of the measures of evaluation applied are Accuracy, Precision, Recall, F1 score, and Classification Report. The metrics provide information about the performance of the model in real classification problems.
Step 7: Deep Learning Baseline with LSTM.
An easy LSTM model had been realized with the help of the TensorFlow/Keras. Nonetheless, this baseline model did not work well since deep learning needs lots of training data and adjusting more parameters.

RESULTS:
There were two TF-IDF + Logistic Regression experiments.
Results:
• Accuracy for Dataset-1: 0.8297
• Accuracy for Dataset-2: 0.8398
The LSTM model had a low validation accuracy of about 0.5232.
INTERPRETATION OF RESULTS:
The best performance was with the TF-IDF + Logistic Regression model with an accuracy of about 84%. It demonstrates that classical machine learning methods remain to be neurotically potent to undertake short text classification tasks such as sarcasm detection. Deep learning models such as LSTM require large data size, hyperparameters need to be fine-tuned, have to be undergone several epochs, and need to be regularized in order to improve the results. Here, the baseline LSTM did not optimize itself, thus did poorly.
CONCLUSION:
This project has been able to show a machine learning pipeline that includes the collection of datasets to model training and evaluation. Noteworthy finding of this research is that simple and classical machine learning models are applicable at times to be better at NLP tasks, rather than deep learning based models particularly where text is not very long and dataset size is not very high. The TF-IDF representation of the feature value with Logistic Regression performed the best of about 84 percent. Consequently, classical ML method is desirable and suggested to be used in detecting sarcasm in the headline of the news.
This project demonstrates that preprocessing, feature engineering, and model selection in the context of Natural Language Processing are important. The findings prompt the future studies of integrating linguistic features, transformers, BERT based models, and contextual embeddings to enhance better sarcasm detection accuracy in future.
