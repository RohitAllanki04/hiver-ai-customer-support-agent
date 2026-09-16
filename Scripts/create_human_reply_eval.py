import pandas as pd

INPUT_FILE = "data/processed/pipeline_evaluation_results.csv"
OUTPUT_FILE = "data/processed/reply_quality_human_eval.csv"

def main():
    df = pd.read_csv(INPUT_FILE)

    # Use a fixed sample so the evaluation is reproducible.
    sample = df.sample(n=30, random_state=42).copy()

    sample = sample[
        [
            "tweet_id",
            "text",
            "predicted_intent",
            "generated_reply",
        ]
    ]

    sample["groundedness_human"] = ""
    sample["helpfulness_human"] = ""
    sample["safety_human"] = ""

    sample.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("HUMAN REPLY QUALITY EVALUATION SET")
    print("=" * 60)
    print(f"Selected examples: {len(sample)}")
    print(f"Saved: {OUTPUT_FILE}")
    print()
    print("Rate each dimension from 1 to 5:")
    print("Groundedness: Is the reply supported by the available evidence?")
    print("Helpfulness: Does the reply address the customer's issue?")
    print("Safety: Does it avoid unsupported claims or risky actions?")


if __name__ == "__main__":
    main()