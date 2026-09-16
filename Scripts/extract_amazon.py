import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    r"C:\Users\sujat\.cache\kagglehub\datasets"
    r"\thoughtvector\customer-support-on-twitter"
    r"\versions\10\twcs\twcs.csv"
)

OUTPUT_PATH = "data/processed/amazon_support.csv"

BRAND = "AmazonHelp"

CHUNK_SIZE = 100_000


# ============================================================
# HELPER
# ============================================================

def clean_id(value):
    """
    Convert tweet IDs to a consistent string representation.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    # Protect against IDs accidentally becoming 12345.0
    if value.endswith(".0"):
        value = value[:-2]

    return value


# ============================================================
# STEP 1
# COLLECT AMAZONHELP TWEET IDS
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — COLLECTING AMAZONHELP TWEET IDS")
print("=" * 70)

amazon_tweet_ids = set()

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
        ],
        dtype={
            "tweet_id": "string",
            "author_id": "string",
            "inbound": "boolean",
        },
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    print(f"Processing chunk {chunk_number}...")

    amazon_rows = chunk[
        (chunk["author_id"] == BRAND)
        &
        (chunk["inbound"] == False)
    ]

    for tweet_id in amazon_rows["tweet_id"]:

        tweet_id = clean_id(tweet_id)

        if tweet_id is not None:
            amazon_tweet_ids.add(tweet_id)


print(
    f"\nAmazonHelp support tweets found: "
    f"{len(amazon_tweet_ids):,}"
)


# ============================================================
# STEP 2
# EXTRACT AMAZONHELP + CUSTOMER MESSAGES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — EXTRACTING AMAZONHELP CONVERSATIONS")
print("=" * 70)

selected_chunks = []

amazon_count = 0
customer_count = 0


for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
        dtype={
            "tweet_id": "string",
            "author_id": "string",
            "inbound": "boolean",
            "created_at": "string",
            "text": "string",
            "response_tweet_id": "string",
            "in_response_to_tweet_id": "string",
        },
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    print(f"Processing chunk {chunk_number}...")

    # --------------------------------------------------------
    # Normalize IDs
    # --------------------------------------------------------

    chunk["tweet_id"] = (
        chunk["tweet_id"]
        .map(clean_id)
    )

    chunk["in_response_to_tweet_id"] = (
        chunk["in_response_to_tweet_id"]
        .map(clean_id)
    )

    # --------------------------------------------------------
    # AmazonHelp messages
    # --------------------------------------------------------

    amazon_rows = chunk[
        chunk["author_id"] == BRAND
    ].copy()

    # --------------------------------------------------------
    # Customer messages that directly reply to AmazonHelp
    # --------------------------------------------------------

    customer_rows = chunk[
        chunk["inbound"] == True
    ].copy()

    customer_rows = customer_rows[
        customer_rows[
            "in_response_to_tweet_id"
        ].isin(amazon_tweet_ids)
    ].copy()

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    amazon_count += len(amazon_rows)

    customer_count += len(customer_rows)

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    selected = pd.concat(
        [
            amazon_rows,
            customer_rows,
        ],
        ignore_index=True,
    )

    if not selected.empty:
        selected_chunks.append(selected)


# ============================================================
# STEP 3
# COMBINE
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — COMBINING RESULTS")
print("=" * 70)

if not selected_chunks:

    raise RuntimeError(
        "No AmazonHelp data was extracted."
    )


amazon_df = pd.concat(
    selected_chunks,
    ignore_index=True,
)


# ============================================================
# STEP 4
# REMOVE DUPLICATE TWEETS
# ============================================================

amazon_df = amazon_df.drop_duplicates(
    subset=["tweet_id"]
)


# ============================================================
# STEP 5
# SORT BY CREATED TIME
# ============================================================

amazon_df["created_at"] = pd.to_datetime(
    amazon_df["created_at"],
    errors="coerce",
    format="mixed",
)

amazon_df = amazon_df.sort_values(
    "created_at"
).reset_index(drop=True)


# ============================================================
# STEP 6
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

amazon_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# STEP 7
# FINAL SUMMARY
# ============================================================

actual_customer_count = (
    amazon_df["inbound"] == True
).sum()

actual_amazon_count = (
    amazon_df["inbound"] == False
).sum()


print("\n" + "=" * 70)
print("AMAZONHELP EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"\nTotal extracted tweets: "
    f"{len(amazon_df):,}"
)

print(
    f"Customer tweets: "
    f"{actual_customer_count:,}"
)

print(
    f"AmazonHelp tweets: "
    f"{actual_amazon_count:,}"
)

print(
    f"\nSaved to:"
    f"\n{OUTPUT_PATH}"
)