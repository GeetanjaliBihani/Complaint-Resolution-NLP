# Complaint-Resolution-NLP

An automated complaint resolution system, which analyzes the text from a given complaint and recommends top three solutions.

The recommendation model architecture is divided into three parts:
1. Text data preprocessing
2. Feature extraction from text data
3. Multi-class classification to predict recommended solutions


![alt text](https://geetanjalibihani.files.wordpress.com/2020/01/1.jpg)

Thus, once a complaint is logged into the system, the text undergoes preprocessing and several features are extracted, including TFIDF features, prevalent topic model theme and cosine similarity scores between complaints and follow-ups. A multi-class classifier, trained on the historical text data along with the engineered features, outputs top three recommended solutions.
