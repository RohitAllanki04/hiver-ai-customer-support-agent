# Golden Evaluation Set

## Purpose

The golden evaluation set is used as the held-out evaluation set for the
Amazon customer-support intent classification pipeline.

It contains 150 customer-support messages covering 13 intent categories.

## Sampling

The examples were sampled from the Amazon customer-support portion of the
Customer Support on Twitter dataset.

The sampling process:

1. Selected inbound customer messages.
2. Removed empty messages and duplicate message text.
3. Excluded examples already present in the initial evaluation set.
4. Applied a minimum message-length filter when constructing the additional
   candidate pool to reduce extremely short or ambiguous examples.
5. Sampled a larger candidate pool and organized candidates by the existing
   classifier's predicted intent to make manual review more efficient.
6. Selected examples covering all supported intent categories.

The classifier predictions were used for candidate organization and
pre-screening; they were not treated as ground-truth labels.

## Labeling

The final intent labels were manually reviewed against the project's
13-intent taxonomy.

The final golden set contains 150 labelled examples with no missing labels
and no duplicate tweet IDs.

## Intent Distribution

| Intent | Examples |
|---|---:|
| delivery_issue | 27 |
| other_support | 19 |
| order_cancellation | 11 |
| order_status | 11 |
| payment_billing | 11 |
| prime_membership | 11 |
| seller_marketplace | 11 |
| refund | 11 |
| product_issue | 9 |
| product_availability_pricing | 9 |
| account_access | 8 |
| delivery_address_options | 6 |
| return_replacement | 6 |
| **Total** | **150** |

## Evaluation

The golden set is evaluated using accuracy and macro F1.

The same 150 examples are used to compare:

- Majority-class baseline
- TF-IDF + Logistic Regression
- Embedding-based intent classifier

The golden set is not used to train these models.