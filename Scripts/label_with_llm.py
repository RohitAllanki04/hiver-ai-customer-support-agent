import os
import time
import pandas as pd

from dotenv import load_dotenv
from groq import Groq


# --------------------------------------------------
# 1. Load API key
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)


# --------------------------------------------------
# 2. Intent definitions
# --------------------------------------------------

INTENTS = {
    "order_status": "Questions about order status, tracking, or when an order is expected.",

    "delivery_issue": "Problems with delivery such as delays, failed delivery, missing package, wrong delivery status, or carrier problems.",

    "delivery_address_options": "Requests or problems related to delivery address, delivery location, delivery time, or delivery options.",

    "order_cancellation": "Requests or problems related to cancelling an order.",

    "return_replacement": "Requests or problems related to returning an item or getting a replacement.",

    "refund": "Questions or problems related to refunds, including refund status, missing refunds, or refund amount.",

    "product_issue": "Problems with the received product, such as wrong, damaged, defective, missing, or poor-quality items.",

    "payment_billing": "Problems or questions about payments, charges, billing, cards, financing, or transaction amounts.",

    "account_access": "Problems with Amazon account access, login, password, account blocking, or account settings.",

    "prime_membership": "Questions or problems related to Amazon Prime membership, benefits, subscription, or Prime-specific services.",

    "seller_marketplace": "Problems specifically involving third-party sellers or marketplace orders.",

    "product_availability_pricing": "Questions about product availability, release dates, pre-orders, prices, discounts, or promotions.",

    "installation_service": "Problems involving installation, appointments, or other scheduled Amazon services.",

    "other_support": "A genuine Amazon support request that does not clearly fit another intent."
}


intent_text = "\n".join(
    f"- {name}: {description}"
    for name, description in INTENTS.items()
)


# --------------------------------------------------
# 3. LLM classification function
# --------------------------------------------------

def classify_message(message):

    prompt = f"""
You are labeling customer-support messages for AmazonHelp.

Choose exactly ONE primary intent from the allowed intents below.

IMPORTANT RULES:

1. Return exactly one intent name.
2. Do not create a new intent.
3. Choose the customer's MAIN reason for contacting support.
4. Do not classify based only on keywords.
5. If a message mentions Prime but the actual problem is delivery,
   choose delivery_issue.
6. If a message mentions an order but the main request is cancellation,
   choose order_cancellation.
7. If a message involves a third-party seller or marketplace,
   choose seller_marketplace.
8. If the message does not clearly fit another category,
   choose other_support.

ALLOWED INTENTS:

{intent_text}

CUSTOMER MESSAGE:

{message}

Return ONLY the intent name.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# --------------------------------------------------
# 4. Batch labeling
# --------------------------------------------------

if __name__ == "__main__":

    input_file = "data/processed/amazon_labeling_candidates.csv"
    output_file = "data/processed/amazon_llm_labels.csv"

    df = pd.read_csv(input_file)

    print(f"Candidate messages: {len(df)}")

    # ----------------------------------------------
    # Resume support
    # ----------------------------------------------

    if os.path.exists(output_file):

        results = pd.read_csv(output_file)

        processed_ids = set(
            results["tweet_id"].astype(str)
        )

        print(
            f"Already processed: {len(processed_ids)}"
        )

    else:

        results = pd.DataFrame(
            columns=[
                "tweet_id",
                "created_at",
                "text",
                "llm_intent"
            ]
        )

        processed_ids = set()


    new_results = []


    # ----------------------------------------------
    # Process messages
    # ----------------------------------------------

    for index, row in df.iterrows():

        tweet_id = str(row["tweet_id"])

        if tweet_id in processed_ids:
            continue

        text = str(row["text"])


        try:

            predicted_intent = classify_message(text)


            # --------------------------------------
            # Validate LLM response
            # --------------------------------------

            if predicted_intent not in INTENTS:

                print(
                    f"Invalid intent for {tweet_id}: "
                    f"{predicted_intent}"
                )

                predicted_intent = "other_support"


            new_results.append(
                {
                    "tweet_id": tweet_id,
                    "created_at": row["created_at"],
                    "text": text,
                    "llm_intent": predicted_intent
                }
            )


            print(
                f"[{index + 1}/{len(df)}] "
                f"{predicted_intent}"
            )


        except Exception as e:

            print(
                f"Error processing tweet "
                f"{tweet_id}: {e}"
            )


        # ------------------------------------------
        # Save every 25 successful labels
        # ------------------------------------------

        if len(new_results) >= 25:

            batch_df = pd.DataFrame(new_results)

            results = pd.concat(
                [results, batch_df],
                ignore_index=True
            )

            results.to_csv(
                output_file,
                index=False
            )

            processed_ids.update(
                batch_df["tweet_id"].astype(str)
            )

            new_results = []

            print(
                f"Progress saved: {len(results)} messages"
            )


        time.sleep(0.2)


    # ----------------------------------------------
    # Save remaining results
    # ----------------------------------------------

    if new_results:

        batch_df = pd.DataFrame(new_results)

        results = pd.concat(
            [results, batch_df],
            ignore_index=True
        )

        results.to_csv(
            output_file,
            index=False
        )


    print("\nLabeling completed.")

    print(
        f"Total labeled: {len(results)}"
    )

    print(
        f"Saved to: {output_file}"
    )


    print("\nIntent distribution:")

    print(
        results["llm_intent"].value_counts()
    )