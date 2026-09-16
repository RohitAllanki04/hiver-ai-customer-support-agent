from src.support_agent import SupportAgent


agent = SupportAgent()


test_messages = [
    "Where is my package? It was supposed to arrive yesterday.",
    "I want to cancel my order before it ships.",
    "I haven't received my refund yet.",
    "I can't log into my Amazon account.",
    "My card was charged twice for the same order."
]


for message in test_messages:

    print("\n" + "=" * 70)
    print("SUPPORT AGENT RESULT")
    print("=" * 70)

    result = agent.generate_reply(message)

    print("\nCustomer message:")
    print(result["customer_message"])

    print("\nPredicted intent:")
    print(result["intent"])

    print("\nGenerated reply:")
    print(result["reply"])

    print("\nResponse valid:")
    print(result["valid"])

    print("\nValidation issues:")
    print(result["validation_issues"])

    print("\nRetrieved examples:")

    for i, example in enumerate(result["retrieved_examples"], start=1):
        print(f"\n--- Example {i} ---")
        print(f"Similarity: {example['similarity']:.4f}")
        print(f"Customer: {example['customer_message']}")
        print(f"AmazonHelp: {example['amazon_response']}")