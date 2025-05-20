import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(page_title="Fake News Classifier", layout="wide")

# Title and description
st.title("Fake News Classifier")
st.write("This app analyzes a news dataset and trains a logistic regression model to classify news as Fake or Real.")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file (must contain 'text' and 'label' columns)", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        
        # Basic EDA
        st.subheader("Exploratory Data Analysis")
        st.write("Dataset Shape:", df.shape)
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        st.write("Class Distribution:")
        st.write(df['label'].value_counts())

        # Check for required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            st.error("Error: Dataset must contain 'text' and 'label' columns.")
        else:
            # Ensure 'text' column is string type and handle missing values
            df['text'] = df['text'].astype(str).fillna('')
            df['label'] = df['label'].fillna(-1)  # Placeholder for invalid labels
            if not df['label'].isin([0, 1]).all():
                st.error("Error: 'label' column must contain only 0 or 1 values.")
            else:
                # Text preprocessing and model training
                X = df['text']
                y = df['label']  # Assuming 'label' is 0 for fake, 1 for real or vice versa

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # TF-IDF Vectorization
                vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
                X_train_tfidf = vectorizer.fit_transform(X_train)
                X_test_tfidf = vectorizer.transform(X_test)

                # Logistic Regression model
                model = LogisticRegression()
                model.fit(X_train_tfidf, y_train)
                y_pred = model.predict(X_test_tfidf)

                # Evaluation
                st.subheader("Model Evaluation")
                st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'], ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)

                # Optional: Predict on user input
                st.subheader("Try It Yourself")
                user_input = st.text_area("Enter a news article text to classify:", height=200)
                if st.button("Classify"):
                    if user_input:
                        # Transform user input
                        user_tfidf = vectorizer.transform([user_input])
                        prediction = model.predict(user_tfidf)[0]
                        label = 'Real' if prediction == 1 else 'Fake'
                        st.write(f"Prediction: **{label}**")
                    else:
                        st.warning("Please enter some text to classify.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
else:
    st.info("Please upload a CSV file to proceed.")
