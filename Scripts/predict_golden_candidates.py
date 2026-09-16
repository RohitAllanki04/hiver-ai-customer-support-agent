import pandas as pd

from src.intent_classifier import IntentClassifier


INPUT_FILE = "data/processed/amazon_golden_candidates_500.csv"
OUTPUT_FILE = "data/processed/amazon_golden_candidates_500_predicted.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    classifier = IntentClassifier(
        classifier_path="models/embedding_intent_classifier.joblib"
    )

    df["predicted_intent"] = df["text"].apply(
        classifier.predict
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 60)
    print("GOLDEN CANDIDATE PRE-SCREENING")
    print("=" * 60)

    print(f"Candidates: {len(df)}")
    print(f"Output: {OUTPUT_FILE}")

    print()
    print("Predicted intent distribution:")
    print(
        df["predicted_intent"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()