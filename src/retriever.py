import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


class SupportRetriever:

    def __init__(
        self,
        retrieval_data_path,
        embeddings_path,
        model_name="all-MiniLM-L6-v2",
        top_k=5
    ):
        self.top_k = top_k

        # Load historical customer → AmazonHelp pairs
        self.retrieval_df = pd.read_csv(retrieval_data_path)

        # Load saved embeddings
        self.embeddings = np.load(embeddings_path)

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Build similarity index
        self.index = NearestNeighbors(
            n_neighbors=top_k,
            metric="cosine",
            algorithm="brute"
        )

        self.index.fit(self.embeddings)

    def retrieve(self, query):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        distances, indices = self.index.kneighbors(query_embedding)

        results = []

        for distance, idx in zip(distances[0], indices[0]):
            row = self.retrieval_df.iloc[idx]

            results.append({
                "similarity": float(1 - distance),
                "customer_message": row["customer_message"],
                "amazon_response": row["amazon_response"]
            })

        return results