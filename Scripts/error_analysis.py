import pandas as pd


INPUT_FILE = "data/processed/pipeline_evaluation_results.csv"
OUTPUT_FILE = "data/processed/intent_error_analysis.csv"


def main():
    print("=" * 60)
    print("HIVER SUPPORT AGENT - INTENT ERROR ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    # Keep only incorrect classifications
    errors = df[
        df["predicted_intent"] != df["human_intent"]
    ].copy()

    # Select useful columns
    errors = errors[
        [
            "tweet_id",
            "text",
            "predicted_intent",
            "human_intent",
        ]
    ]

    # Add a confusion pair
    errors["confusion_pair"] = (
        errors["human_intent"]
        + " -> "
        + errors["predicted_intent"]
    )

    # Count how frequently each confusion occurs
    confusion_counts = (
        errors["confusion_pair"]
        .value_counts()
        .rename("count")
    )

    errors["confusion_count"] = (
        errors["confusion_pair"]
        .map(confusion_counts)
    )

    # Sort most common errors first
    errors = errors.sort_values(
        by=["confusion_count", "human_intent"],
        ascending=[False, True]
    )

    # Save detailed error analysis
    errors.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Total evaluation samples: {len(df)}")
    print(f"Incorrect classifications: {len(errors)}")
    print(
        f"Error rate: "
        f"{len(errors) / len(df):.2%}"
    )

    print()
    print("TOP CONFUSION PAIRS")
    print("-" * 60)

    for pair, count in confusion_counts.items():
        print(f"{pair}: {count}")

    print()
    print("ERROR ANALYSIS FILE")
    print("-" * 60)
    print(OUTPUT_FILE)

    print()
    print("ERROR ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()