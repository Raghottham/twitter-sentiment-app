import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load trimmed dataset
column_names = ['Target', 'ID', 'Date', 'Flag', 'User', 'Text']
twitter_data = pd.read_csv('tiny_dataset.csv', names=column_names, encoding='ISO-8859-1')

# Convert target values: 4 → 1 (positive), 0 stays 0 (negative)
twitter_data.replace({'Target': {4: 1}}, inplace=True)

# Stemming function
port_stem = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# Apply stemming
twitter_data['stemmed_content'] = twitter_data['Text'].apply(stemming)

# Prepare input and labels
X = twitter_data['stemmed_content'].values
Y = twitter_data['Target'].values

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Accuracy (optional)
train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))
print("✅ Training Accuracy:", train_acc)
print("✅ Testing Accuracy:", test_acc)

# Save model and vectorizer
pickle.dump(model, open('trained_model.sav', 'wb'))
pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))

print("🎉 Model and vectorizer saved successfully!")
