import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

email_texts = [
    "Get rich quick! Win a million dollars now!",
    "Meeting tomorrow at 10 AM",
    "You've won a free gift card!",
    "Reminder: Don't forget your appointment",
    "Claim your prize today!"
]

# Corresponding labels ("spam" or "ham")
labels = ["spam", "ham", "spam", "ham", "spam"]

# df = pd.read_csv('spam.csv')
# df1 = pd.read_csv('email_dataset.csv')
# df2 = pd.read_csv('emails.csv')


# df['label'] = df['label'].apply(lambda x: 0 if x == 'ham' else 1)

# df1['label'] = df1['label'].apply(lambda x: 0 if x == 'ham' else 1)
# df2['label'] = df2['label'].apply(lambda x: 0 if x == 'ham' else 1)

# data = pd.concat([df], axis=0)

# Extract email texts and labels from the DataFrame
# email_texts = data['text'].tolist()
# labels = data['label'].tolist()

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(email_texts, labels, test_size=0.2, random_state=42)

# Tokenize and prepare data for the model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Encode the emails into input features
X_train_encoded = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt", max_length=128)
X_test_encoded = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt", max_length=128)

# Create DataLoader for training and testing datasets
train_dataset = TensorDataset(X_train_encoded["input_ids"], X_train_encoded["attention_mask"], torch.tensor(y_train))
test_dataset = TensorDataset(X_test_encoded["input_ids"], X_test_encoded["attention_mask"], torch.tensor(y_test))

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids, attention_mask, labels = (item.to(device) for item in batch)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Avg. Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = (item.to(device) for item in batch)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Decode labels
predicted_labels = label_encoder.inverse_transform(all_preds)
true_labels = label_encoder.inverse_transform(all_labels)

# Print classification report
print(classification_report(true_labels, predicted_labels))
