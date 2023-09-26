import klib as klib
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('spam.csv')
df1 = pd.read_csv('email_dataset.csv')
df2 = pd.read_csv('emails.csv')


df['label'] = df['label'].apply(lambda x: 0 if x == 'ham' else 1)

df1['label'] = df1['label'].apply(lambda x: 0 if x == 'ham' else 1)
df2['label'] = df2['label'].apply(lambda x: 0 if x == 'ham' else 1)

df_concat = pd.concat([df, df1, df2], axis=0)

# Save the DataFrame to a CSV file
#df.to_csv('email_classification_data.csv', index=False)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_concat['text'], df_concat['label'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer for text feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Fit and transform the vectorizer on the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train an XGBoost classifier
model = xgb.XGBClassifier()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

new_email = ["Congratulation on your wedding bro"]
new_email_counts = vectorizer.transform(new_email)
predicted_class = classifier.predict(new_email_counts)[0]
predicted_class = "ham" if predicted_class ==0 else "spam"
print(f"Predicted class for new email: {predicted_class}")