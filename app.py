# from flask import Flask, request, jsonify
# import joblib
# import re
# import string
# import nltk
# from nltk.corpus import stopwords

# # Initialize Flask app
# app = Flask(__name__)

# # Load the model and vectorizer
# model = joblib.load('model.pkl')
# vectorizer = joblib.load('vectorizer.pkl')

# # Initialize NLTK components
# nltk.download('stopwords')
# stopword = set(stopwords.words('english'))
# stemmer = nltk.SnowballStemmer("english")

# # Text cleaning function
# def clean(text):
#     text = str(text).lower()
#     text = re.sub('\[.*?\]', '', text)
#     text = re.sub('https?://\S+|www\.\S+', '', text)
#     text = re.sub('<.*?>+', '', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub('\n', '', text)
#     text = re.sub('\w*\d\w*', '', text)
#     text = [word for word in text.split(' ') if word not in stopword]
#     text = " ".join(text)
#     text = [stemmer.stem(word) for word in text.split(' ')]
#     text = " ".join(text)
#     return text

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     text = data['text']
#     cleaned_text = clean(text)
#     vectorized_text = vectorizer.transform([cleaned_text])
#     prediction = model.predict(vectorized_text)
#     return jsonify({'prediction': prediction[0]})

# if __name__ == '__main__':
#     app.run(port=5000)






from flask import Flask, request, jsonify
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Initialize NLTK components
nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

# Load model and vectorizer
model = None
vectorizer = None

def load_model_and_vectorizer():
    global model, vectorizer
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

# Define the text cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        load_model_and_vectorizer()
    
    data = request.json
    text = data.get('text', '')
    cleaned_text = clean(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return jsonify({'prediction': prediction[0]})

@app.route('/')
def index():
    return "Flask API is running!"

if __name__ == '__main__':
    app.run(port=5000)



