import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from textblob import TextBlob
from newspaper import Article

# Load the data
data = pd.read_csv(r"I:\learing_projects\fakeNewsDetecter\fake_or_real_news.csv")

# Preprocess the data
data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)
data = data.drop('label', axis=1)

# Split the data
X, y = data['text'], data['fake']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)

# Transform the test data using the fitted vectorizer
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
clf = LinearSVC(random_state=42)
clf.fit(X_train_vectorized, y_train)


url = 'https://www.cnbc.com/2024/07/15/cnbc-daily-open-trump-assassination-attempt.html'

article = Article(url)
article.download()
article.parse()
article.nlp()

articleText = article.text
articleSummaryVectorized = vectorizer.transform([articleText])

prediction = clf.predict(articleSummaryVectorized)[0]
probability = clf.decision_function(articleSummaryVectorized)[0]

if prediction == 1:
    print(f'The model thinks the article is fake (confidence: {abs(probability):.2f})')
else:
    print(f'The model thinks the article is real (confidence: {abs(probability):.2f})')

print(f'\nArticle Summary:\n{article.summary}')
print(f'\nArticle Keywords: {", ".join(article.keywords)}')


"""
# Evaluate the model
train_score = clf.score(X_train_vectorized, y_train)
test_score = clf.score(X_test_vectorized, y_test)

print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Generate predictions
y_pred = clf.predict(X_test_vectorized)

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
"""
