from src.intent_classifier import IntentClassifier
from src.retriever import SupportRetriever
from src.generator import SupportResponseGenerator
from src.response_validator import ResponseValidator


class SupportAgent:

    def __init__(self):
        # Intent classifier
        self.classifier = IntentClassifier(
            classifier_path="models/embedding_intent_classifier.joblib"
        )

        # Semantic retriever
        self.retriever = SupportRetriever(
            retrieval_data_path="data/processed/amazon_retrieval_pairs.csv",
            embeddings_path="models/retrieval_embeddings.npy",
            top_k=5
        )

        # LLM response generator
        self.generator = SupportResponseGenerator()

        # Response safety validator
        self.validator = ResponseValidator()

    def generate_reply(self, customer_message):

        # 1. Classify intent
        intent = self.classifier.predict(customer_message)

        # 2. Retrieve similar historical conversations
        retrieved_examples = self.retriever.retrieve(
            customer_message
        )

        # 3. Generate response using LLM
        reply = self.generator.generate(
            customer_message=customer_message,
            intent=intent,
            retrieved_examples=retrieved_examples
        )

        # 4. Validate generated response
        validation = self.validator.validate(
            response=reply,
            customer_message=customer_message
        )

        # 5. Return complete result
        return {
            "customer_message": customer_message,
            "intent": intent,
            "retrieved_examples": retrieved_examples,
            "reply": validation["response"],
            "valid": validation["valid"],
            "validation_issues": validation["issues"]
        }