# Decision Log

This document records non-obvious engineering and evaluation decisions made
during development of the Hiver customer-support AI agent.

## 1. Selected AmazonHelp as the target brand

**Decision:** Build the support agent for AmazonHelp.

**Why:** Amazon has a large volume of customer-support conversations in the
provided dataset, providing enough examples for intent classification,
retrieval, and response generation.

---

## 2. Defined a 13-intent taxonomy

**Decision:** Use 13 support intents instead of attempting to reproduce every
possible topic in the dataset.

**Intents:**

- `account_access`
- `delivery_address_options`
- `delivery_issue`
- `order_cancellation`
- `order_status`
- `other_support`
- `payment_billing`
- `prime_membership`
- `product_availability_pricing`
- `product_issue`
- `refund`
- `return_replacement`
- `seller_marketplace`

**Why:** A smaller, explicit taxonomy makes classification and evaluation
tractable while still covering the major support categories observed in the
selected Amazon data.

---

## 3. Used an embedding-based classifier

**Decision:** Use sentence embeddings with a supervised classifier for intent
prediction.

**Why:** Customer-support messages often express the same intent using very
different wording. Semantic embeddings provide a representation based on
meaning rather than relying only on exact keywords.

---

## 4. Used a majority-class baseline

**Decision:** Include a majority-class classifier as the trivial baseline.

**Why:** It establishes the minimum performance obtained by always predicting
the most frequent intent and provides context for interpreting model accuracy.

---

## 5. Used TF-IDF + Logistic Regression as the simple baseline

**Decision:** Compare the embedding classifier against TF-IDF features with
Logistic Regression.

**Why:** TF-IDF + Logistic Regression is a simple, reproducible text
classification baseline that tests whether the additional semantic
representation provides measurable improvement.

---

## 6. Created a 150-example golden evaluation set

**Decision:** Use 150 labelled examples as the final golden evaluation set.

**Why:** The assignment requires a 150–250 example hand-labelled evaluation
set. A 150-example set provides coverage across all 13 intents while keeping
manual review practical.

---

## 7. Used classifier predictions only to organize candidate review

**Decision:** Existing classifier predictions were used to organize and
pre-screen candidate examples, but were not treated as ground-truth labels.

**Why:** This made it easier to find examples across all intent categories
while keeping the final intent label separate from the model prediction.

**Caveat:** Because model predictions influenced candidate selection, the
sampling process can introduce selection bias. This is documented rather
than hidden.

---

## 8. Removed duplicate message text during candidate sampling

**Decision:** Remove duplicate customer-message text when constructing
candidate examples.

**Why:** Duplicate messages would provide less information during evaluation
and could make the evaluation set appear larger without increasing example
diversity.

---

## 9. Used top-5 retrieval examples

**Decision:** Retrieve the five most similar historical support examples for
each customer message.

**Why:** A small retrieval set provides enough historical context for response
generation without unnecessarily increasing prompt size or introducing too
many weakly related examples.

---

## 10. Grounded generation on historical support conversations

**Decision:** Give the generator retrieved historical customer-support
conversations as evidence.

**Why:** The assignment requires replies to be grounded in the brand's
historical resolutions. Retrieval provides concrete examples of how similar
issues were handled.

---

## 11. Prohibited unsupported operational claims

**Decision:** The generator is instructed not to invent refunds, replacements,
escalations, account actions, tracking information, policies, or specific
resolution timelines.

**Why:** Historical examples provide guidance but do not prove that the same
action has been performed for the current customer. Preventing unsupported
claims reduces the risk of misleading customers.

---

## 12. Added a response-validation stage

**Decision:** Validate the generated response after LLM generation.

**Why:** Generation quality cannot be assumed from the LLM output alone.
Validation provides an additional check before the response is returned by
the support agent.

---

## 13. Added auto-handle versus escalation decisions

**Decision:** Add an explicit decision layer that returns either
`auto_handle` or `escalate`, together with a reason.

**Why:** The assignment requires the agent to decide whether a request can be
handled automatically or should be escalated to human support.

The initial policy escalates requests that may require account-specific or
transaction-specific actions, including account access, payment/billing,
refund, and order cancellation. Responses that fail validation are also
escalated.

---

## 14. Increased generation token limit to 400

**Decision:** Increase the generator's maximum output tokens from 300 to 400.

**Why:** Earlier testing produced responses that were truncated before the
customer-facing sentence was complete. Increasing the limit reduced this
failure mode while keeping responses concise.

---

## 15. Evaluated the LLM judge against reviewer reference ratings

**Decision:** Evaluate the LLM-as-judge against reviewer reference ratings
rather than treating the judge's scores as ground truth.

**Why:** LLM judges can disagree with human/reviewer assessments. Measuring
exact agreement, quadratic weighted kappa, and mean absolute error makes this
limitation visible.

The current 30-example evaluation produced:

- Groundedness: 40.0% exact agreement, quadratic weighted κ = 0.443
- Helpfulness: 36.7% exact agreement, quadratic weighted κ = 0.614
- Safety: 60.0% exact agreement, quadratic weighted κ = 0.325

These results are reported as evidence about judge alignment rather than as
proof that the judge is perfectly reliable.

---

## Summary

The system deliberately separates:

1. **Intent classification**
2. **Historical evidence retrieval**
3. **Response generation**
4. **Response validation**
5. **Escalation decision**
6. **Evaluation**

This separation makes it possible to measure individual components and
identify where failures occur instead of relying only on an end-to-end
headline number.