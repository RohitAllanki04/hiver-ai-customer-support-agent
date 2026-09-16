from src.support_agent import SupportAgent


# Initialize the complete support agent
agent = SupportAgent()


# Test customer message
message = "Where is my package? It was supposed to arrive yesterday."


# Run the complete pipeline
result = agent.generate_reply(message)


print("\n=== SUPPORT AGENT ===")
print("Customer message:", result["customer_message"])
print("Predicted intent:", result["intent"])


print("\n=== RETRIEVED EXAMPLES ===")

for i, example in enumerate(result["retrieved_examples"], start=1):
    print(f"\n--- Example {i} ---")
    print(f"Similarity: {example['similarity']:.4f}")
    print(f"Customer: {example['customer_message']}")
    print(f"AmazonHelp: {example['amazon_response']}")


print("\n=== GENERATED RESPONSE ===")
print(result["reply"])


print("\n=== VALIDATION ===")
print("Valid:", result["valid"])
print("Issues:", result["validation_issues"])