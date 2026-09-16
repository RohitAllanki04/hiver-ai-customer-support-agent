from src.intent_classifier import IntentClassifier
from src.retriever import SupportRetriever
from src.generator import SupportResponseGenerator
from src.response_validator import ResponseValidator


class SupportAgent:
    def __init__(self):
        self.classifier = IntentClassifier(
            classifier_path="models/embedding_intent_classifier.joblib"
        )

        self.retriever = SupportRetriever(
            retrieval_data_path="data/processed/amazon_retrieval_pairs.csv",
            embeddings_path="models/retrieval_embeddings.npy",
            top_k=5
        )

        self.generator = SupportResponseGenerator()
        self.validator = ResponseValidator()

    def decide_escalation(self, intent, validation):
        """
        Decide whether the request should be handled automatically
        or escalated to human support.
        """

        # If the generated response fails validation,
        # do not automatically send it to the customer.
        if not validation["valid"]:
            return {
                "decision": "escalate",
                "reason": (
                    "The generated response failed safety or quality "
                    "validation."
                )
            }

        # These intents may require account-specific or
        # transaction-specific actions that the agent cannot perform.
        escalation_intents = {
            "account_access",
            "payment_billing",
            "refund",
            "order_cancellation"
        }

        if intent in escalation_intents:
            return {
                "decision": "escalate",
                "reason": (
                    f"The request may require account or "
                    f"transaction-specific action for the "
                    f"'{intent}' issue."
                )
            }

        return {
            "decision": "auto_handle",
            "reason": (
                "The request can be addressed using available "
                "historical support evidence without account-specific action."
            )
        }

    def generate_reply(self, customer_message):
        # 1. Classify customer intent
        intent = self.classifier.predict(customer_message)

        # 2. Retrieve similar historical support conversations
        retrieved_examples = self.retriever.retrieve(customer_message)

        # 3. Generate customer-facing response
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

        # 5. Decide auto-handle vs human escalation
        escalation = self.decide_escalation(
            intent=intent,
            validation=validation
        )

        # 6. Return complete pipeline result
        return {
            "customer_message": customer_message,
            "intent": intent,
            "retrieved_examples": retrieved_examples,
            "reply": validation["response"],
            "valid": validation["valid"],
            "validation_issues": validation["issues"],
            "decision": escalation["decision"],
            "decision_reason": escalation["reason"]
        }