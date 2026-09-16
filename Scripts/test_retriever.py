from src.retriever import SupportRetriever


retriever = SupportRetriever(
    retrieval_data_path="data/processed/amazon_retrieval_pairs.csv",
    embeddings_path="models/retrieval_embeddings.npy",
    top_k=5
)


query = "Where is my package? It was supposed to arrive yesterday."

results = retriever.retrieve(query)


for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Customer: {result['customer_message']}")
    print(f"AmazonHelp: {result['amazon_response']}")