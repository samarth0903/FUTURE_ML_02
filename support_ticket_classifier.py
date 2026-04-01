# ==========================================
# Support Ticket Classification System
# Machine Learning Task 2 – Future Interns
# ==========================================

import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn as sns
import matplotlib.pyplot as plt


# ------------------------------------------
# Download NLTK resources
# ------------------------------------------

nltk.download("punkt")
nltk.download("stopwords")


# ------------------------------------------
# 1 Load Dataset
# ------------------------------------------

data = pd.read_csv("all_tickets_processed_improved_v3.csv")

print("\nDataset Loaded Successfully")
print(data.head())

print("\nDataset Shape:")
print(data.shape)


# ------------------------------------------
# 2 Text Cleaning
# ------------------------------------------

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = str(text).lower()

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word.isalpha()]

    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


print("\nCleaning Text...")

data["clean_text"] = data["Document"].apply(clean_text)


# ------------------------------------------
# 3 Feature Extraction (TF-IDF)
# ------------------------------------------

print("\nConverting Text to TF-IDF Features...")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["clean_text"])

y = data["Topic_group"]


# ------------------------------------------
# 4 Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ------------------------------------------
# 5 Train Classification Model
# ------------------------------------------

print("\nTraining Model...")

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)


# ------------------------------------------
# 6 Predictions
# ------------------------------------------

predictions = model.predict(X_test)


# ------------------------------------------
# 7 Model Evaluation
# ------------------------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions, average="weighted")

recall = recall_score(y_test, predictions, average="weighted")


print("\n========== Model Evaluation ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)


print("\nClassification Report:")
print(classification_report(y_test, predictions))


# ------------------------------------------
# 8 Confusion Matrix
# ------------------------------------------

print("\nGenerating Confusion Matrix...")

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(10,7))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# ------------------------------------------
# 9 Priority Assignment
# ------------------------------------------

def assign_priority(category):

    if category in ["Hardware", "Access"]:
        return "High"

    elif category in ["Purchase", "Administrative rights"]:
        return "Medium"

    else:
        return "Low"


data["priority"] = data["Topic_group"].apply(assign_priority)


print("\nTicket Priority Examples:")

print(data[["Document","Topic_group","priority"]].head())


# ------------------------------------------
# 10 Save Output
# ------------------------------------------

data.to_csv("ticket_predictions.csv", index=False)

print("\nPredictions saved to: ticket_predictions.csv")


# ------------------------------------------
# 11 Business Insight
# ------------------------------------------

print("\nBusiness Insight:")

print("This system automatically categorizes support tickets and assigns priorities.")
print("High priority tickets can be handled immediately by support teams.")
print("This reduces response time and improves customer satisfaction.")