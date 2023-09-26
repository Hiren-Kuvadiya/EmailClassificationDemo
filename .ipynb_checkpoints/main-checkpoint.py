# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample email dataset (replace this with your own labeled dataset)
emails = [
    ("spam", "Buy cheap watches now!"),
    ("spam", "Win a free iPhone!"),
    ("ham", "Hey, how's it going?"),
    ("ham", "Meeting tomorrow at 2 PM."),
    # Add more emails with labels as needed
]

# Separate emails and labels
labels, texts = zip(*emails)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text into numerical features
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_counts)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report for more detailed metrics
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Example usage: Classify a new email
new_email = ["Congratulations! You've won $1000 in a lottery."]
new_email_counts = vectorizer.transform(new_email)
predicted_class = classifier.predict(new_email_counts)[0]
print(f"Predicted class for new email: {predicted_class}")

