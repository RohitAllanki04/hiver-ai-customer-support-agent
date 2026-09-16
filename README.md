
## Hiver - AI Customer Support Agent

An AI-powered customer support system that understands customer messages, finds similar past support conversations, generates a helpful reply, and checks the reply before returning it.

This project was built for the Hiver SDE Intern take-home assignment.

---

## What does this project do?

Imagine a customer sends Amazon a message:

> "Where is my package? It was supposed to arrive yesterday."

Instead of manually handling every message, this system processes the message through several steps:

```text
Customer Message
       |
       v
Understand the problem
       |
       v
Find similar past conversations
       |
       v
Generate a helpful reply
       |
       v
Check whether the reply is safe
       |
       v
Return the final response
````

For the example above, the system can identify the issue as:

```text
delivery_issue
```

It then looks at similar historical customer-support conversations and uses them as guidance when generating the response.

---

## Why is this useful?

Customer-support teams receive large numbers of messages every day.

Many messages are about similar problems:

* Where is my order?
* I want to cancel my order.
* I haven't received my refund.
* I cannot access my account.
* My payment was charged twice.
* The product I received is damaged.

Instead of treating every message as completely new, this system uses previous support conversations to understand the type of problem and help generate a relevant response.

The goal is not to blindly copy an old response.

The system uses historical responses as **examples and guidance** while generating a new response for the current customer.

---

# How the System Works

The system has four main stages.

```text
1. Intent Classification
          |
          v
2. Similar Conversation Retrieval
          |
          v
3. Response Generation
          |
          v
4. Response Validation
```

## 1. Intent Classification

First, the system tries to understand what the customer is asking about.

For example:

| Customer message                | Predicted intent     |
| ------------------------------- | -------------------- |
| "Where is my package?"          | `delivery_issue`     |
| "I want to cancel my order."    | `order_cancellation` |
| "I haven't received my refund." | `refund`             |
| "I can't log into my account."  | `account_access`     |
| "My card was charged twice."    | `payment_billing`    |

The classifier currently supports 13 intents:

* `account_access`
* `delivery_address_options`
* `delivery_issue`
* `order_cancellation`
* `order_status`
* `other_support`
* `payment_billing`
* `prime_membership`
* `product_availability_pricing`
* `product_issue`
* `refund`
* `return_replacement`
* `seller_marketplace`

The trained classifier is stored at:

```text
models/embedding_intent_classifier.joblib
```

---

# 2. Similar Conversation Retrieval

After identifying the customer's intent, the system searches through historical Amazon customer-support conversations.

For example, if the customer says:

> "Where is my package? It was supposed to arrive yesterday."

the system can find previous conversations about missing or delayed deliveries.

The retriever returns the **5 most similar historical conversations**.

Example:

```text
Customer:
"Where is my package? It was supposed to arrive yesterday."

Similar historical conversation:
"It says delivered yesterday at 10pm. I was at home and no one
delivered anything. Where is my package?"

Historical response:
"Have you checked with close neighbours and safe places
around the property?"
```

These historical conversations are **not directly copied into the final response**.

They are given to the language model as examples of how similar support issues were handled.

Retrieval data:

```text
data/processed/amazon_retrieval_pairs.csv
```

Precomputed embeddings:

```text
models/retrieval_embeddings.npy
```

---

# 3. Response Generation

The retrieved examples, customer message, and predicted intent are given to an LLM.

The LLM generates a new response specifically for the customer's message.

For example:

```text
Customer message:

"Where is my package? It was supposed to arrive yesterday."
```

The system may generate:

```text
I'm sorry you haven't received your package yet. Please
double-check any safe-drop locations or with nearby neighbors,
and let us know which carrier is listed for the order so we can
look into it further. If you need immediate assistance, you can
contact Amazon Support.
```

The response generator uses Groq.

Current model:

```text
openai/gpt-oss-120b
```

The generator is designed to avoid:

* Making up order information.
* Making up refund information.
* Claiming that it accessed a customer's account.
* Copying URLs from historical conversations.
* Copying personal identifiers.
* Providing unsupported contact information.
* Returning unnecessary information.

The historical responses are used as **guidance**, not as templates to copy.

---

# 4. Response Validation

The generated response is checked before it is returned to the customer.

The validator checks whether the response violates the safety and response constraints defined for the project.

For example, the system should not generate a response that:

```text
- Contains a customer's private information
- Copies a historical URL
- Claims access to the customer's account
- Invents unsupported order information
- Returns an empty or incomplete response
```

Only after validation does the system return the final response.

---

# Complete Architecture

```text
                    Customer Message
                           |
                           v
                +---------------------+
                |  Intent Classifier  |
                |                     |
                | Understands the     |
                | type of problem     |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Conversation        |
                | Retriever           |
                |                     |
                | Finds similar past  |
                | support messages    |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Response Generator  |
                |                     |
                | LLM creates a new   |
                | customer response   |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Response Validator  |
                |                     |
                | Checks the response |
                +----------+----------+
                           |
                           v
                    Final Response
```

---

# Dataset

The project uses the **Customer Support on Twitter** dataset.

The dataset contains real-world-style customer-support conversations between customers and brands on Twitter.

For this project, Amazon-related conversations were extracted and processed for:

* Intent labeling
* Intent classification
* Conversation retrieval
* Response generation
* Evaluation

The processed data is stored under:

```text
data/processed/
```

Important files include:

```text
amazon_clean.csv
amazon_conversations.csv
amazon_training_data.csv
amazon_retrieval_pairs.csv
amazon_final_evaluation_104_filled.csv
intent_error_analysis.csv
pipeline_evaluation_results.csv
```

---

# Evaluation

The system was evaluated on:

```text
104 manually labeled customer-support messages
```

The evaluation measures how accurately the system identifies the customer's intent and whether the generated responses pass validation.

## Results

| Metric                |   Result |
| --------------------- | -------: |
| Evaluation samples    |      104 |
| Intent accuracy       |   70.19% |
| Incorrect predictions | 31 / 104 |
| Response validation   |     100% |
| Pipeline errors       |        0 |

The intent classifier achieved:

```text
70.19% accuracy
```

The response validation stage successfully validated:

```text
104 / 104 responses
```

The complete evaluation run had:

```text
0 pipeline errors
```

---

# Intent Classification Results

| Intent                       | Precision | Recall |   F1 |
| ---------------------------- | --------: | -----: | ---: |
| account_access               |      0.50 |   1.00 | 0.67 |
| delivery_address_options     |      0.38 |   0.75 | 0.50 |
| delivery_issue               |      1.00 |   0.38 | 0.55 |
| order_cancellation           |      0.88 |   1.00 | 0.93 |
| order_status                 |      0.88 |   1.00 | 0.93 |
| other_support                |      0.62 |   0.28 | 0.38 |
| payment_billing              |      0.88 |   1.00 | 0.93 |
| prime_membership             |      0.88 |   1.00 | 0.93 |
| product_availability_pricing |      0.50 |   0.67 | 0.57 |
| product_issue                |      0.62 |   0.83 | 0.71 |
| refund                       |      0.88 |   1.00 | 0.93 |
| return_replacement           |      0.38 |   1.00 | 0.55 |
| seller_marketplace           |      0.75 |   0.86 | 0.80 |

---

# Error Analysis

The system made:

```text
31 incorrect predictions out of 104
```

Error analysis was performed to understand where the classifier has difficulty.

Some common confusion patterns were:

```text
other_support -> account_access
delivery_issue -> delivery_address_options
other_support -> return_replacement
delivery_issue -> product_availability_pricing
delivery_issue -> other_support
```

For example, some customers write very general messages without enough information to clearly determine whether the issue is related to an account, delivery, returns, or another support category.

The complete error analysis is available at:

```text
data/processed/intent_error_analysis.csv
```

This analysis can be used to improve the intent definitions and training data in future iterations.

---

# API

The project exposes the support agent through a FastAPI application.

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

---

## Generate a Support Response

```http
POST /api/v1/support/reply
```

### Input

```json
{
  "message": "Where is my order?"
}
```

### What happens internally?

```text
"Where is my order?"
        |
        v
Intent Classification
        |
        v
order_status
        |
        v
Find similar conversations
        |
        v
Generate response
        |
        v
Validate response
```

### Example output

```json
{
  "message": "Where is my order?",
  "intent": "order_status",
  "reply": "I can help with your order status. Please check your order details for the latest delivery information.",
  "valid": true,
  "validation_issues": []
}
```

The actual generated response can vary because it is generated by an LLM.

---

# Running the Project

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Hiver
```

---

## 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 4. Configure the API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git using `.gitignore`.

**Never commit your API key to GitHub.**

---

## 5. Start the FastAPI application

```powershell
uvicorn app.main:app --reload
```

The API will start at:

```text
http://127.0.0.1:8000
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

You can use the `/docs` page to send test messages directly to the API.

---

# Testing

The project contains several test scripts.

Available tests:

```text
Scripts/
├── test_generator.py
├── test_intent_classifier.py
├── test_retriever.py
├── test_support_agent.py
└── test_support_pipeline.py
```

Because the project uses the `src` package, run the tests from the project root using Python's module syntax.

## Test the Intent Classifier

```powershell
python -m Scripts.test_intent_classifier
```

Example:

```text
Message: Where is my package?
Predicted intent: delivery_issue

Message: I want to cancel my order.
Predicted intent: order_cancellation

Message: I haven't received my refund yet.
Predicted intent: refund
```

---

## Test the Retriever

```powershell
python -m Scripts.test_retriever
```

This tests whether the system can find similar historical support conversations.

---

## Test the Response Generator

```powershell
python -m Scripts.test_generator
```

This tests the LLM response generation component.

---

## Test the Complete Support Agent

```powershell
python -m Scripts.test_support_agent
```

This tests the complete flow:

```text
Customer Message
      |
      v
Intent Classification
      |
      v
Retrieval
      |
      v
Response Generation
      |
      v
Response Validation
```

---

## Test the Support Pipeline

```powershell
python -m Scripts.test_support_pipeline
```

This provides an end-to-end smoke test of the support agent.

---

# Running the Evaluation

To evaluate the complete pipeline:

```powershell
python Scripts/evaluate_pipeline.py
```

The evaluation produces:

```text
data/processed/pipeline_evaluation_results.csv
```

To perform intent error analysis:

```powershell
python Scripts/error_analysis.py
```

The resulting analysis is stored at:

```text
data/processed/intent_error_analysis.csv
```

---

# Project Structure

```text
Hiver/
|
+-- app/
|   +-- main.py
|
+-- data/
|   +-- raw/
|   +-- processed/
|
+-- models/
|   +-- embedding_intent_classifier.joblib
|   +-- intent_classifier.joblib
|   +-- retrieval_embeddings.npy
|
+-- notebooks/
|   +-- 01_data_exploration.ipynb
|   +-- 02_intent_exploration.ipynb
|   +-- 03_validate_llm_labels.ipynb
|   +-- 04_prepare_training_data.ipynb
|   +-- 05_train_intent_classifier.ipynb
|   +-- 06_embedding_intent_classifier.ipynb
|   +-- 07_conversation_retrieval.ipynb
|
+-- Scripts/
|   +-- analyze_conversations.py
|   +-- clean_amazon_data.py
|   +-- download_dataset.py
|   +-- error_analysis.py
|   +-- evaluate_pipeline.py
|   +-- extract_amazon.py
|   +-- label_with_llm.py
|   +-- test_generator.py
|   +-- test_intent_classifier.py
|   +-- test_retriever.py
|   +-- test_support_agent.py
|   +-- test_support_pipeline.py
|
+-- src/
|   +-- generator.py
|   +-- intent_classifier.py
|   +-- response_validator.py
|   +-- retriever.py
|   +-- support_agent.py
|
+-- tests/
|
+-- .gitignore
+-- README.md
+-- requirements.txt
```

---

# Technology Stack

### Backend

* Python
* FastAPI
* Pydantic

### Machine Learning

* Scikit-learn
* Sentence Transformers
* Joblib
* NumPy

### Data Processing

* Pandas
* NumPy

### Retrieval

* Sentence embeddings
* Nearest-neighbor similarity search

### LLM

* Groq
* `openai/gpt-oss-120b`

### Development

* Git
* GitHub
* Jupyter Notebooks

---

# Important Design Decisions

## Historical conversations are examples, not templates

The system retrieves previous conversations to help the LLM understand how similar problems were handled.

It does not simply return the historical response.

This allows the system to generate a new response based on:

```text
Current customer message
        +
Predicted intent
        +
Similar historical conversations
```

---

## Response safety

Historical support conversations can contain information such as:

* URLs
* Usernames
* Personal references
* Social media handles

The response generator is explicitly instructed not to copy these details into the generated response.

This helps prevent accidental reproduction of information from historical conversations.

---

# Limitations

The current system has several limitations.

### Intent classification

The current intent accuracy is:

```text
70.19%
```

The evaluation set contains 104 manually labeled messages, so this result should be interpreted in the context of that evaluation set.

Some intent categories have overlapping meanings, especially broad support and delivery-related categories.

### Response generation

The LLM generates responses dynamically, so wording can vary between requests.

### Account-specific information

The system does not have access to a customer's Amazon account, order history, payment information, or delivery status.

Therefore, it cannot independently verify account-specific information.

When account-specific information is required, the system directs the customer toward Amazon Support rather than pretending to access their account.

---

# Future Improvements

Possible improvements include:

* Add more manually labeled training examples.
* Improve the definition of overlapping intents.
* Use the error analysis results to improve classifier training.
* Add confidence scores to intent predictions.
* Add a fallback for low-confidence predictions.
* Improve retrieval using intent-aware filtering.
* Add automated regression tests for known classification errors.
* Improve response-quality evaluation using human reviewers.
* Add structured logging.
* Add monitoring for production deployment.

---

# End-to-End Summary

The complete system can be summarized as:

```text
Customer Message
       |
       v
Understand the customer's intent
       |
       v
Find similar historical conversations
       |
       v
Use those examples as guidance
       |
       v
Generate a new response with an LLM
       |
       v
Validate the response
       |
       v
Return the customer-facing response
```

The system was evaluated on 104 manually labeled messages and achieved:

```text
Intent Accuracy:     70.19%
Response Validation: 100%
Pipeline Errors:     0
```

The project also includes error analysis to identify where the intent classifier needs further improvement.



