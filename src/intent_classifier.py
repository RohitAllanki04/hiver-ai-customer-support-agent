import joblib
from sentence_transformers import SentenceTransformer


class IntentClassifier:

    def __init__(
        self,
        classifier_path="models/embedding_intent_classifier.joblib",
        model_name="all-MiniLM-L6-v2"
    ):
        # Load trained classifier
        self.classifier = joblib.load(classifier_path)

        # Load embedding model
        self.model = SentenceTransformer(model_name)

    def predict(self, text):
        # Convert message to embedding
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Predict intent
        intent = self.classifier.predict(embedding)[0]

        return intent