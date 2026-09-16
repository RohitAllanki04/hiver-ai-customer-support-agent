from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# Project setup
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

# Allow imports from project root
sys.path.insert(0, str(ROOT_DIR))

from src.support_agent import SupportAgent


# ============================================================
# Configuration
# ============================================================

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "amazon_final_evaluation_104_filled.csv"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "pipeline_evaluation_results.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("HIVER SUPPORT AGENT - PIPELINE EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load evaluation dataset
    # --------------------------------------------------------

    print("\nLoading evaluation dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset: {DATA_PATH}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # --------------------------------------------------------
    # 2. Check required columns
    # --------------------------------------------------------

    required_columns = [
        "text",
        "human_intent"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 3. Remove rows without human labels
    # --------------------------------------------------------

    df = df.dropna(
        subset=["text", "human_intent"]
    ).copy()

    print(
        f"Rows used for evaluation: {len(df)}"
    )

    if len(df) == 0:
        raise ValueError(
            "No labeled rows available for evaluation."
        )

    # --------------------------------------------------------
    # 4. Load the SAME SupportAgent used by FastAPI
    # --------------------------------------------------------

    print("\nLoading SupportAgent...")

    agent = SupportAgent()

    print("SupportAgent loaded successfully.")

    # --------------------------------------------------------
    # 5. Run complete pipeline
    # --------------------------------------------------------

    predicted_intents = []
    generated_replies = []
    reply_valid = []
    validation_issues = []

    print("\nRunning pipeline evaluation...")
    print("-" * 60)

    total = len(df)

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        message = str(row["text"])

        try:

            result = agent.generate_reply(
                message
            )

            predicted_intents.append(
                result["intent"]
            )

            generated_replies.append(
                result["reply"]
            )

            reply_valid.append(
                bool(result["valid"])
            )

            issues = result.get(
                "validation_issues",
                []
            )

            validation_issues.append(
                "; ".join(map(str, issues))
            )

            print(
                f"[{i}/{total}] "
                f"intent={result['intent']} "
                f"| valid={result['valid']}"
            )

        except Exception as e:

            print(
                f"[{i}/{total}] ERROR: {e}"
            )

            predicted_intents.append(
                "ERROR"
            )

            generated_replies.append(
                ""
            )

            reply_valid.append(
                False
            )

            validation_issues.append(
                str(e)
            )

    # --------------------------------------------------------
    # 6. Add predictions to dataframe
    # --------------------------------------------------------

    df["predicted_intent"] = (
        predicted_intents
    )

    df["generated_reply"] = (
        generated_replies
    )

    df["reply_valid"] = (
        reply_valid
    )

    df["validation_issues"] = (
        validation_issues
    )

    # --------------------------------------------------------
    # 7. Intent classification evaluation
    # --------------------------------------------------------

    y_true = (
        df["human_intent"]
        .astype(str)
        .str.strip()
    )

    y_pred = (
        df["predicted_intent"]
        .astype(str)
        .str.strip()
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\n")
    print("=" * 60)
    print("INTENT CLASSIFICATION RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")
    print()

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # 8. Response validation evaluation
    # --------------------------------------------------------

    valid_count = int(
        df["reply_valid"].sum()
    )

    total_count = len(df)

    validation_rate = (
        valid_count / total_count
    )

    print("=" * 60)
    print("RESPONSE VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"\nValid responses: "
        f"{valid_count}/{total_count}"
    )

    print(
        f"Validation rate: "
        f"{validation_rate:.4f}"
    )

    # --------------------------------------------------------
    # 9. Pipeline summary
    # --------------------------------------------------------

    error_count = int(
        (df["predicted_intent"] == "ERROR").sum()
    )

    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    print(
        f"\nTotal messages: {total_count}"
    )

    print(
        f"Intent accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"Valid replies: "
        f"{valid_count}/{total_count}"
    )

    print(
        f"Reply validation rate: "
        f"{validation_rate:.2%}"
    )

    print(
        f"Pipeline errors: "
        f"{error_count}"
    )

    # --------------------------------------------------------
    # 10. Save detailed evaluation results
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n")
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"\nDetailed results saved to:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()