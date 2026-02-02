import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download nltk resources (only once)
nltk.download('punkt')
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')

# Define Clean function (list-safe for pipeline)
def Clean(doc):
    """
    Clean a string or a list of strings:
    - remove non-alphabetic characters
    - lowercase
    - remove stopwords
    - lemmatize
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # If input is a list (from FunctionTransformer)
    if isinstance(doc, list):
        cleaned_docs = []
        for d in doc:
            d = re.sub(r"[^a-zA-Z\s]", "", d)
            d = d.lower()
            tokens = nltk.word_tokenize(d)
            tokens = [w for w in tokens if w not in stop_words]
            tokens = [lemmatizer.lemmatize(w) for w in tokens]
            cleaned_docs.append(" ".join(tokens))
        return cleaned_docs
    else:
        # Single string
        doc = re.sub(r"[^a-zA-Z\s]", "", doc)
        doc = doc.lower()
        tokens = nltk.word_tokenize(doc)
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)

# Load trained pipeline
with open("pipe_svm.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="Mental Health Text Classifier", layout="centered")
st.title("Mental Health Text Classification")
st.write("Enter text and the model will classify it as **Yes / No**")

# User input
user_text = st.text_area("Enter text here", height=150)

# Predict button
if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Pass raw text to pipeline (pipeline includes FunctionTransformer)
        prediction = model.predict([user_text])[0]  # always wrap in list
        
        if prediction == "YES":
            st.success("✅ YES – Mental health issue detected")
        else:
            st.info("❌ NO – Mental health issue not detected")
