# Research Abstract Classifier

A machine learning project that classifies academic research abstracts into one of three disciplines: **Science**, **Business**, or **Social Science**. The project uses natural language processing (NLP), TF-IDF feature extraction, and compares six traditional machine learning classifiers.

## Project Objectives

- Clean and combine academic abstracts from three Scopus CSV datasets.
- Create a balanced dataset for the three research categories.
- Preprocess text using tokenization, stopword removal, and lemmatization.
- Convert text into numerical features using TF-IDF with unigrams and bigrams.
- Train and evaluate multiple classification models.
- Compare model performance using evaluation metrics and visualizations.
- Demonstrate predictions through an interactive Jupyter Notebook interface.

## Models Compared

The notebook trains and evaluates these six classifiers:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Decision Tree
4. k-Nearest Neighbors (kNN)
5. Support Vector Machine (linear SVM)
6. Random Forest

The notebook calculates **accuracy, precision, recall, and F1-score**, and displays classification reports, confusion matrices, and an accuracy comparison chart. The model results are generated when the notebook is run.

## Workflow

1. **Load and clean data** — Read the three CSV files, retain title and abstract, remove missing values, placeholder abstracts, short abstracts, and duplicate titles/abstracts.
2. **Balance the classes** — Select the same number of rows from each category, shuffle them, and save the combined dataset as `academic_abstracts_dataset.csv`.
3. **Preprocess text** — Lowercase text, remove non-alphabetic characters, tokenize, remove English stopwords and short tokens, apply part-of-speech tagging, and lemmatize words.
4. **Split the dataset** — Encode category labels and create an 80/20 stratified training/test split.
5. **Extract features** — Use TF-IDF with unigrams and bigrams, a maximum of 5,000 features, `min_df=3`, and sublinear term-frequency scaling.
6. **Train and evaluate models** — Fit all six classifiers and compare their metrics.
7. **Visualize results** — Plot confusion matrices and model accuracy.
8. **Try the interactive demo** — Enter a research abstract in the notebook widget to receive a predicted category and confidence scores from the Logistic Regression model.

## Requirements

- Python 3.9 or later is recommended.
- Jupyter Notebook or JupyterLab
- pandas
- NumPy
- scikit-learn
- NLTK
- Matplotlib
- Seaborn
- ipywidgets

## How to Run

1. Clone or download this repository.
2. Put `scopus_science.csv`, `scopus_business.csv`, and `scopus_social.csv` in the same directory as the notebook.
3. Install the requirements listed above.
4. Start Jupyter
5. Open `Shivam_kumar_26565014_research_classification_project.ipynb`.
6. Run the notebook cells **from top to bottom** so that the dataset, preprocessing function, vectorizer, models, and evaluation results are created in the correct order.
7. In the **Interactive Demo** section, paste a research abstract and click **Predict Category**.

If the interactive widgets do not appear, ensure `ipywidgets` is installed and enabled for your Jupyter environment.

## Evaluation

The notebook reports the following metrics:

- **Accuracy:** proportion of test examples classified correctly.
- **Precision:** proportion of predicted examples for a class that are correct.
- **Recall:** proportion of actual examples for a class that are found.
- **F1-score:** harmonic mean of precision and recall.

## Results and Performance
All models were evaluated using Accuracy, Precision, Recall, and F1-Measure (macro-averaged to ensure equal weighting across all three disciplines). 

* Logistic Regression, SVM, and Naive Bayes demonstrated the highest overall performance. The models achieved high baseline accuracy because the core vocabularies of Science, Business, and Social Science have minimal overlap.
* **Interpretation:** Confusion matrices were generated for all six classifiers to visualize exact misclassifications. These matrices successfully highlighted edge cases, such as the mathematical overlap between Business (Marketing) and Social Science (Psychology).

**Final Model Performance Comparison:**

| Model | Accuracy | Precision | Recall | F1-Score | 
| ----- | ----- | ----- | ----- | ----- | 
| **Logistic Regression** | **99.17%** | **0.9917** | **0.9917** | **0.9917** | 
| Support Vector Machine (SVM) | 98.33% | 0.9834 | 0.9833 | 0.9833 | 
| Random Forest | 97.92% | 0.9794 | 0.9792 | 0.9791 | 
| Multinomial Naive Bayes | 97.50% | 0.9753 | 0.9750 | 0.9751 | 
| k-Nearest Neighbors (kNN) | 96.67% | 0.9679 | 0.9667 | 0.9666 | 
| Decision Tree | 96.25% | 0.9627 | 0.9625 | 0.9626 | 

## Future Improvements

- Use random or stratified sampling when balancing the source categories.
- Tune model hyperparameters and evaluate with cross-validation.
- Investigate probability calibration and inspect errors by category.
- Compare TF-IDF models with transformer-based text representations.
- Package the trained preprocessing pipeline and model for reuse outside the notebook.
