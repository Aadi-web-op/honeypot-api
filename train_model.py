import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def main():
    # 1. Load Data
    try:
        df = pd.read_csv("scam_dataset.csv")
    except FileNotFoundError:
        print("Dataset not found. Run generate_data.py first.")
        return

    print(f"Loaded {len(df)} samples.")

    # 2. Vectorize
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # 5. Evaluate
    print("Evaluating Model...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 6. Save Models
    joblib.dump(clf, "scam_classifier.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Models saved: scam_classifier.pkl, tfidf_vectorizer.pkl")

if __name__ == "__main__":
    main()
