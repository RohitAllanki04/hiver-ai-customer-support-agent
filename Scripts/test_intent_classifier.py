from src.intent_classifier import IntentClassifier


classifier = IntentClassifier(
    classifier_path="models/embedding_intent_classifier.joblib"
)


test_messages = [
    "Where is my package? It was supposed to arrive yesterday.",
    "I want to cancel my order.",
    "I haven't received my refund yet.",
    "I can't log into my Amazon account.",
    "My payment was charged twice."
]


for message in test_messages:
    intent = classifier.predict(message)

    print("\nMessage:", message)
    print("Predicted intent:", intent)