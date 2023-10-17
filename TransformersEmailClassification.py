import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Define the model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Example email text
email_text = "Congratulations! You've won a free vacation. Claim your prize now!"

# Tokenize the email text
inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)

# Perform inference for email classification
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class (0 for spam, 1 for not spam)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

# Define labels for classification
class_labels = ["spam", "not spam"]

# Print the result
print(f"Predicted class: {class_labels[predicted_class]}")
