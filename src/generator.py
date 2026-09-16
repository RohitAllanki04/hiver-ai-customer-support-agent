import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class SupportResponseGenerator:

    def __init__(self, model="openai/gpt-oss-120b"):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(
        self,
        customer_message,
        intent,
        retrieved_examples
    ):

        # --------------------------------------------------
        # Build retrieved examples
        # --------------------------------------------------

        examples_text = ""

        for i, example in enumerate(
            retrieved_examples,
            start=1
        ):
            examples_text += (
                f"\nExample {i}:\n"
                f"Customer: "
                f"{example['customer_message']}\n"
                f"AmazonHelp: "
                f"{example['amazon_response']}\n"
            )

        # --------------------------------------------------
        # Prompt
        # --------------------------------------------------

        prompt = f"""
You are an Amazon customer-support response drafting assistant.

Your task is to draft a concise, helpful support reply to the customer's
message.

Customer message:
{customer_message}

Predicted intent:
{intent}

Here are similar historical AmazonHelp conversations:
{examples_text}

Instructions:
- Address the customer's main issue directly.
- Use the historical responses as guidance, not as text to copy.
- Do not invent order details, refunds, tracking information, policies,
  or actions that are not supported by the available information.
- Do not claim that you accessed the customer's account.
- Do not ask the customer to provide an order number unless the retrieved
  examples clearly support that request.
- Do not generate URLs, links, email addresses, phone numbers, or social
  media handles.
- Do not copy URLs or personal identifiers from the historical examples.
- If account-specific information is required, direct the customer to
  Amazon Support without providing a URL.
- Use the historical responses only as guidance for tone and resolution.
- Do not mention that you are an AI.
- Keep the response concise and professional.
- Return only the draft customer-facing response.
- Your response must contain at least one complete sentence.
"""

        # --------------------------------------------------
        # Call Groq
        # --------------------------------------------------

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You draft safe and helpful "
                        "customer-support replies."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=400
        )

        # --------------------------------------------------
        # Safely extract generated content
        # --------------------------------------------------

        content = response.choices[0].message.content

        # --------------------------------------------------
        # Fallback for None response
        # --------------------------------------------------

        if content is None:
            return (
                "Sorry, we couldn't generate a response "
                "to your request. Please contact Amazon "
                "Support for further assistance."
            )

        content = content.strip()

        # --------------------------------------------------
        # Fallback for empty or incomplete responses
        # --------------------------------------------------

        if not content or len(content) < 20:
            return (
                "Sorry, we couldn't generate a complete response "
                "to your request. Please contact Amazon "
                "Support for further assistance."
            )

        return content