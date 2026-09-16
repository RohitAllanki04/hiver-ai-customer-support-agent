import pandas as pd

from src.retriever import SupportRetriever


INPUT_FILE = "data/processed/reply_quality_human_eval.csv"
OUTPUT_FILE = "data/processed/reply_quality_human_eval_with_evidence.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    retriever = SupportRetriever(
        retrieval_data_path="data/processed/amazon_retrieval_pairs.csv",
        embeddings_path="models/retrieval_embeddings.npy",
        top_k=5,
    )

    evidence = []

    for text in df["text"].astype(str):
        results = retriever.retrieve(text)

        # Keep the historical customer/support examples as text
        formatted = []

        for item in results:
            if isinstance(item, dict):
                formatted.append(str(item))
            else:
                formatted.append(str(item))

        evidence.append("\n---\n".join(formatted))

    df["retrieved_evidence"] = evidence

    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("RETRIEVAL EVIDENCE ADDED")
    print("=" * 60)
    print(f"Examples: {len(df)}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()