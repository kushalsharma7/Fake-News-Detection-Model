# Fake-News-Detection-Modelws Detection System is a machine learning–based web application designed to classify news articles as true or false. The project leverages supervised learning techniques, trained on labeled datasets originating from data.world, containing two primary files: true.csv and fake.csv.

This system aims to address one of the modern challenges of digital journalism—identifying misleading or fabricated news rapidly and accurately. It integrates data preprocessing, model training, evaluation, and web deployment into a unified pipeline using tools such as Jupyter Notebook, VS Code, and Streamlit.

Project Objectives
To build an intelligent classifier capable of differentiating between true and fake news articles.

To preprocess and clean text data using advanced Natural Language Processing (NLP) techniques.

To train and evaluate multiple ML models to determine the most robust architecture.

To deploy the final model as an interactive Streamlit web app for end-user accessibility.

Dataset Description
The dataset was sourced from data.world, comprising:

true.csv: Contains legitimate news articles from credible publishers.

fake.csv: Contains fabricated or misleading articles collected from non-verified sources.

Dataset Attributes
Each CSV file includes:

title: Headline of the news article.

text: Article body.

subject: General topic or category.

date: Publication date.

The combined dataset forms the foundation for training and testing the machine learning model.

Workflow Summary
The project was implemented in several stages, starting with exploratory work in Jupyter Notebook, then transferring the final code to VS Code for application development and deployment.

Data Collection and Integration

Merged true.csv and fake.csv into a single DataFrame.

Added a binary label column: 1 for true news, 0 for fake news.

Data Cleaning and Preprocessing

Removed null and duplicate rows.

Converted text to lowercase.

Tokenized text and removed stopwords.

Applied lemmatization for linguistic normalization.

Feature Extraction (Text Vectorization)

Used TF-IDF Vectorizer to transform text data into numerical features suitable for machine learning algorithms.

Model Building and Training

Tested several models including Logistic Regression, PassiveAggressiveClassifier, and Naive Bayes.

The Logistic Regression model achieved optimal results in terms of precision and recall.

Model Evaluation

Evaluated using accuracy, precision, recall, and F1-score.

Verified through confusion matrix analysis.

Deployment

Saved the trained model as a .pkl file.

Deployed using Streamlit to create an interactive web interface for text input and prediction display.

Tools and Technologies
Python 3.10+

Jupyter Notebook – Model development and experiment environment.

Visual Studio Code – Application integration and final deployment environment.

Streamlit – Web framework for deploying ML models interactively.

NumPy, Pandas, scikit-learn, NLTK – Core libraries for data preprocessing and ML.

Pickle – For model serialization.

File Structure
text
Fake-News-Detection/
│
├── data/
│   ├── true.csv
│   ├── fake.csv
│
├── notebooks/
│   └── data_preprocessing.ipynb
│
├── app/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── app.py
│
├── requirements.txt
├── README.md
└── streamlit_app.py
Example Streamlit Interface
The Streamlit interface enables users to input any news article text to predict authenticity in real time.

Example flow:

Enter the article text in the input box.

Click Predict.

The system displays:

“True News” (for verified content)

“Fake News” (for fabricated content)

Core Streamlit Code Snippet
python
import streamlit as st
import pickle

st.title("Fake News Detection App")

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

user_input = st.text_area("Enter a news article here:")
if st.button("Predict"):
    transformed_input = vectorizer.transform([user_input])
    prediction = model.predict(transformed_input)[0]
    if prediction == 1:
        st.success("This appears to be TRUE news.")
    else:
        st.error("This appears to be FAKE news.")
Model Performance
Metric	Logistic Regression	Naive Bayes	PassiveAggressive
Accuracy	98.4%	96.7%	97.5%
Precision	98.1%	96.2%	97.0%
Recall	98.6%	97.0%	97.3%
F1-score	98.3%	96.6%	97.1%
The Logistic Regression model was selected as the production model due to its consistent high performance and faster inference time.

Results and Insights
High model accuracy indicates a strong ability to detect fake news based on textual patterns.

TF-IDF feature representation proved effective over raw text counts.

Logistic Regression balanced computational efficiency and interpretability.

Deployed Streamlit interface provides a practical demonstration for end users.

Future Improvements
Incorporating transformer-based language models (e.g., BERT, RoBERTa) for deeper contextual understanding.

Expanding dataset with multilingual fake news articles.

Adding visualization dashboards for model explainability (e.g., SHAP plots).

Enabling API-based integration for real-time verification from external applications.

Conclusion
The Fake News Detection System successfully demonstrates how machine learning and NLP can be combined to tackle misinformation spread. Built using Python, trained in Jupyter Notebook, refined in VS Code, and deployed with Streamlit, this project offers an end-to-end implementation pipeline—from raw data collection to real-world web deploym
