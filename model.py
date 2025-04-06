import pandas as pd
import joblib
data = pd.read_csv('data/top_celebs.csv')
data['name'] = data['name'].str.strip()
filtered = data.groupby('name').filter(lambda n: len(n) >= 5)
filtered['name'].unique() # celebs with >= 4 tweets

# Train / Test split
from sklearn.model_selection import train_test_split
X = filtered['tweet'].str.strip().str.lower()
y = filtered['name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Model pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words=None)),  # Converts text to numeric features
    ('clf', LogisticRegression(max_iter=1000))  # Multi-class classification
])

model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Accuracy: {score:.2f}")

def get_model():
    return model

joblib.dump(model, 'tweetlike_model.pkl')
