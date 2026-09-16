import pandas as pd
from sklearn.metrics import cohen_kappa_score


HUMAN_FILE = "data/processed/reply_quality_human_eval_with_evidence.csv"
LLM_FILE = "data/processed/reply_quality_llm_judge.csv"


def main():

    human = pd.read_csv(HUMAN_FILE)
    llm = pd.read_csv(LLM_FILE)

    df = human.merge(
        llm,
        on="tweet_id",
        how="inner"
    )

    print("=" * 60)
    print("HIVER - LLM JUDGE / HUMAN AGREEMENT")
    print("=" * 60)

    print(f"Human ratings: {len(human)}")
    print(f"LLM ratings:   {len(llm)}")
    print(f"Matched rows:  {len(df)}")

    dimensions = [
        ("groundedness", "groundedness_human", "groundedness_llm"),
        ("helpfulness", "helpfulness_human", "helpfulness_llm"),
        ("safety", "safety_human", "safety_llm"),
    ]

    results = []

    for name, human_col, llm_col in dimensions:

        human_scores = df[human_col].astype(int)
        llm_scores = df[llm_col].astype(int)

        exact_agreement = (
            human_scores == llm_scores
        ).mean()

        weighted_kappa = cohen_kappa_score(
            human_scores,
            llm_scores,
            weights="quadratic"
        )

        mean_absolute_error = (
            (human_scores - llm_scores)
            .abs()
            .mean()
        )

        results.append(
            {
                "dimension": name,
                "exact_agreement": exact_agreement,
                "quadratic_weighted_kappa": weighted_kappa,
                "mean_absolute_error": mean_absolute_error,
            }
        )

    results_df = pd.DataFrame(results)

    print()
    print(results_df.to_string(index=False))

    output_file = "data/processed/judge_human_agreement.csv"

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print(f"Saved: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()