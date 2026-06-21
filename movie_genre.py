import pandas as pd
import re
import string
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


print("=" * 50)
print("MOVIE GENRE CLASSIFICATION PROJECT")
print("=" * 50)

df = pd.read_csv(
    "movie_genre.csv",
    names=["plot", "genre"]
)

print("\nDataset Loaded Successfully")


print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nGenre Distribution:")
print(df["genre"].value_counts())

plt.figure(figsize=(8,5))

df["genre"].value_counts().plot(
    kind="bar"
)

plt.title("Genre Distribution")
plt.xlabel("Genre")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

print("\nSupported Genres:")
for genre in sorted(df["genre"].unique()):
    print("-", genre)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )
    return text

df["plot"] = df["plot"].apply(clean_text)


X = df["plot"]
y = df["genre"]


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X)

print("\nText Vectorization Completed")



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


model = LogisticRegression(
    max_iter=5000,
    C=10,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed")


y_pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

joblib.dump(
    model,
    "movie_genre_model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print("\nModel Saved Successfully")


while True:

    print("\n" + "-" * 50)

    text = input(
        "Enter Movie Plot (or quit): "
    )

    if text.lower() == "quit":
        print("\nProgram Ended Successfully")
        break

    cleaned_text = clean_text(text)

    data = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(data)

    probabilities = model.predict_proba(
        data
    )

    confidence = max(
        probabilities[0]
    ) * 100

    print("\nPrediction Result")
    print("=" * 25)

    print(
        "Predicted Genre:",
        prediction[0]
    )

    print(
        "Confidence:",
        round(confidence, 2),
        "%"
    )

    print("=" * 25)