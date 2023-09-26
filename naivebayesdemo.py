import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Step 1: Define the email data
df = pd.read_csv('spam.csv')

# Step 3: Create a TF-IDF vectorizer for feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Step 4: Fit and transform the vectorizer on the email data
X_tfidf = tfidf_vectorizer.fit_transform(df['text'])

# Step 5: Initialize a custom label encoder with explicit class labels
label_mapping = {'ham': 0, 'spam': 1}
df['label'] = df['label'].map(label_mapping)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['label'], test_size=0.2, random_state=42)

# Step 7: Initialize and train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 8: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

