# Project – I: Research Abstract Classification
**Date:** October 25, 2026

## 1. Problem Statement
The objective of this project is to build a machine learning classification system capable of categorizing peer-reviewed academic abstracts into three distinct fields: Science, Business, and Social Science. This baseline system analyzes the textual data within publication abstracts to automatically predict the appropriate disciplinary domain based on vocabulary and context.

## 2. Dataset and Preprocessing
The dataset consists of over 1,000 research abstracts sourced ethically from academic databases. The data was strictly balanced across all three target categories to prevent class imbalance biases during model training. 
* **Text Preprocessing:** The abstracts were cleaned using standard Natural Language Processing (NLP) techniques, including lowercasing, punctuation removal, deduplication, and stop-word filtering.
* **Vectorization:** The text was converted into mathematical representations using TF-IDF (Term Frequency-Inverse Document Frequency). We utilized `sublinear_tf=True` to dampen abnormally high-frequency words and an n-gram range of `(1,2)` to capture highly contextual bigrams (e.g., "social media", "neural network") rather than just isolated words.

## 3. Models Evaluated
Six distinct machine learning classifiers were trained and evaluated on the dataset:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Decision Tree
4. k-Nearest Neighbors (kNN)
5. Support Vector Machine (SVM)
6. Random Forest

## 4. Results and Performance
All models were evaluated using Accuracy, Precision, Recall, and F1-Measure (macro-averaged to ensure equal weighting across all three disciplines). 
* Logistic Regression, SVM, and Naive Bayes demonstrated the highest overall performance. The models achieved high baseline accuracy because the core vocabularies of Science, Business, and Social Science have minimal overlap.
* **Interpretation:** Confusion matrices were generated for all six classifiers to visualize exact misclassifications. These matrices successfully highlighted edge cases, such as the mathematical overlap between Business (Marketing) and Social Science (Psychology) when analyzing human behavior.

## 5. Interactive Demo
A user-friendly, interactive demo was built directly into the final cell of the Jupyter Notebook using Gradio. Users can paste any unclassified abstract into the text interface and instantly receive a predicted label. The demo uses the Logistic Regression model to also provide a visual breakdown of confidence scores (e.g., 85% Business, 10% Social Science, 5% Science), allowing users to see exactly how the model interprets ambiguous text.