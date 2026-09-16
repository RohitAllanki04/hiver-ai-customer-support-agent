import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


DATA_FILE = "data/processed/amazon_golden_evaluation_150.csv"


def majority_baseline(y_train, y_test):
    majority_class = y_train.value_counts().idxmax()
    predictions = [majority_class] * len(y_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
    }


def tfidf_baseline(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


def main():
    print("=" * 60)
    print("HIVER SUPPORT AGENT - BASELINE EVALUATION")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)

    X = df["text"].astype(str)
    y = df["human_intent"].astype(str)

    # Keep the evaluation set separate from baseline training.
    # We train the simple baseline on the original training data
    # and evaluate on the 150-example golden set.
    train_data = pd.read_csv("data/processed/amazon_training_data.csv")

    X_train = train_data["text"].astype(str)
    y_train = train_data["intent"].astype(str)

    print(f"Training samples: {len(train_data)}")
    print(f"Golden evaluation samples: {len(df)}")

    print("\nRunning majority baseline...")
    majority = majority_baseline(y_train, y)

    print("\nRunning TF-IDF + Logistic Regression...")
    tfidf = tfidf_baseline(X_train, X, y_train, y)

    results = pd.DataFrame(
        [
            {
                "model": "Majority baseline",
                "accuracy": majority["accuracy"],
                "macro_f1": majority["macro_f1"],
            },
            {
                "model": "TF-IDF + Logistic Regression",
                "accuracy": tfidf["accuracy"],
                "macro_f1": tfidf["macro_f1"],
            },
            {
                "model": "Embedding classifier",
                "accuracy": df["correct"].mean(),
                "macro_f1": f1_score(
                    y,
                    df["predicted_intent"],
                    average="macro",
                    zero_division=0,
                ),
            },
        ]
    )

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))

    results.to_csv(
        "data/processed/baseline_results.csv",
        index=False,
    )

    print("\nSaved:")
    print("data/processed/baseline_results.csv")


if __name__ == "__main__":
    main()