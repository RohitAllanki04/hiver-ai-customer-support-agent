from src.generator import SupportResponseGenerator


generator = SupportResponseGenerator()


customer_message = (
    "Where is my package? It was supposed to arrive yesterday."
)

intent = "delivery_issue"

retrieved_examples = [
    {
        "customer_message": (
            "@AmazonHelp It says it was delivered yesterday at 10pm. "
            "I was at home and no one delivered anything?!!! Where is my package"
        ),
        "amazon_response": (
            "@324180 Hi, have you checked with close neighbours "
            "&amp; safe places around property?"
        ),
    },
    {
        "customer_message": (
            "@AmazonHelp I was expecting a package on Saturday "
            "it still hasn't come"
        ),
        "amazon_response": (
            "@737514 I'm sorry for the wait! Just to confirm, "
            "which carrier is assigned to the order?"
        ),
    },
]


reply = generator.generate(
    customer_message=customer_message,
    intent=intent,
    retrieved_examples=retrieved_examples
)


print("\n=== GENERATED RESPONSE ===")
print(reply)