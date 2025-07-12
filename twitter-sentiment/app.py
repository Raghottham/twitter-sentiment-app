from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))

# Text preprocessing
stemmer = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    tweet = ""

    if request.method == 'POST':
        tweet = request.form['tweet']
        processed = preprocess_text(tweet)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative'

    return render_template('index.html', sentiment=sentiment, tweet=tweet)

if __name__ == '__main__':
    app.run(debug=True)
