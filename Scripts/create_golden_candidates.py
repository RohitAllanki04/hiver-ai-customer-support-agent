import pandas as pd

INPUT_FILE = "data/processed/amazon_support.csv"
EVAL_FILE = "data/processed/amazon_final_evaluation_104_filled.csv"
OUTPUT_FILE = "data/processed/amazon_golden_candidates_100.csv"


def main():
    df = pd.read_csv(INPUT_FILE)
    existing = pd.read_csv(EVAL_FILE)

    # Keep customer messages only
    candidates = df[
        (df["inbound"] == True) &
        (df["text"].notna())
    ].copy()

    # Remove empty messages
    candidates["text"] = candidates["text"].astype(str).str.strip()
    candidates = candidates[candidates["text"] != ""]

    # Remove tweets already present in the 104-example evaluation set
    used_ids = set(existing["tweet_id"].astype(str))

    candidates = candidates[
        ~candidates["tweet_id"].astype(str).isin(used_ids)
    ]

    # Remove duplicate text
    candidates = candidates.drop_duplicates(
        subset=["text"]
    )

    # Randomly sample 100 candidates
    candidates = candidates.sample(
        n=100,
        random_state=42
    )

    candidates = candidates[
        [
            "tweet_id",
            "created_at",
            "text"
        ]
    ].copy()

    # Empty column for manual labelling
    candidates["human_intent"] = ""

    candidates.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 60)
    print("GOLDEN SET CANDIDATE GENERATION")
    print("=" * 60)
    print(f"Candidates created: {len(candidates)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()