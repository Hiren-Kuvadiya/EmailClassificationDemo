# Import the necessary libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Define your labeled email dataset
# Example: emails, labels = ["spam email 1", "ham email 2"], [1, 0] where 1 is spam, 0 is ham

# Split the dataset into a training set and a test set

emails = [
    "Claim your free vacation today!",
    "Hi, when are we meeting for lunch?",
    "Exclusive discount for you, don't miss out!",
    "Don't forget our meeting at 2 PM.",
]

labels = [1, 0, 1, 0]
df = pd.read_csv('spam.csv')
# Step 3: Create a TF-IDF vectorizer for feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Step 4: Fit and transform the vectorizer on the email data
X_tfidf = tfidf_vectorizer.fit_transform(df['text'])

# Step 5: Initialize a custom label encoder with explicit class labels
label_mapping = {'ham': 0, 'spam': 1}
df['label'] = df['label'].map(label_mapping)


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a CountVectorizer to convert emails into feature vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize an SVM classifier (SVC)
classifier = SVC(kernel='linear')

# Train the SVM classifier on the training data
classifier.fit(X_train_vec, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy
print("Accuracy:", accuracy)


# New email samples to predict
new_emails = [
    "You've won a free gift!",
    "Can we schedule a call tomorrow?",
    "Limited time offer - 50% discount!",
]

# Convert new email samples into feature vectors
new_emails_vec = vectorizer.transform(new_emails)

# Make predictions on the new email samples
predictions = classifier.predict(new_emails_vec)

# Print the predictions
for email, prediction in zip(new_emails, predictions):
    if prediction == 1:
        label = "spam"
    else:
        label = "ham"
    print(f"Email: {email} | Predicted Label: {label}")
