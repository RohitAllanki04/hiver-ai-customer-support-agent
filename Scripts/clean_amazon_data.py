import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/processed/amazon_support.csv"

OUTPUT_PATH = (
    "data/processed/amazon_clean.csv"
)

MIN_TEXT_LENGTH = 5


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — LOADING AMAZONHELP DATA")
print("=" * 70)

df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "tweet_id": "string",
        "author_id": "string",
        "inbound": "boolean",
        "created_at": "string",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string",
    }
)

initial_count = len(df)

print(
    f"\nInitial tweets: {initial_count:,}"
)


# ============================================================
# STEP 2 — REMOVE MISSING TWEET IDS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — REMOVING INVALID TWEET IDS")
print("=" * 70)

missing_tweet_id = (
    df["tweet_id"].isna()
)

print(
    f"Missing tweet IDs: "
    f"{missing_tweet_id.sum():,}"
)

df = df[
    ~missing_tweet_id
].copy()


# ============================================================
# STEP 3 — REMOVE EMPTY TEXT
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — REMOVING EMPTY MESSAGES")
print("=" * 70)

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

empty_text = (
    df["text"] == ""
)

print(
    f"Empty messages: "
    f"{empty_text.sum():,}"
)

df = df[
    ~empty_text
].copy()


# ============================================================
# STEP 4 — REMOVE EXTREMELY SHORT MESSAGES
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — REMOVING EXTREMELY SHORT MESSAGES")
print("=" * 70)

short_text = (
    df["text"].str.len()
    < MIN_TEXT_LENGTH
)

print(
    f"Messages shorter than "
    f"{MIN_TEXT_LENGTH} characters: "
    f"{short_text.sum():,}"
)

df = df[
    ~short_text
].copy()


# ============================================================
# STEP 5 — REMOVE DUPLICATE TWEET IDS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — REMOVING DUPLICATE TWEETS")
print("=" * 70)

duplicate_ids = (
    df["tweet_id"].duplicated()
)

print(
    f"Duplicate tweet IDs: "
    f"{duplicate_ids.sum():,}"
)

df = df[
    ~duplicate_ids
].copy()


# ============================================================
# STEP 6 — CHECK INBOUND VALUES
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — VALIDATING MESSAGE ROLES")
print("=" * 70)

invalid_inbound = (
    df["inbound"].isna()
)

print(
    f"Messages with unknown inbound value: "
    f"{invalid_inbound.sum():,}"
)

df = df[
    ~invalid_inbound
].copy()


# ============================================================
# STEP 7 — PARSE TIMESTAMP
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 — VALIDATING TIMESTAMPS")
print("=" * 70)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    errors="coerce",
    format="mixed",
)

invalid_dates = (
    df["created_at"].isna()
)

print(
    f"Invalid timestamps: "
    f"{invalid_dates.sum():,}"
)

# We keep the rows for now.
# The timestamp itself is not required for intent
# classification, and deleting the conversation could
# remove useful text.
#
# We will use valid timestamps when ordering conversations.


# ============================================================
# STEP 8 — NORMALIZE WHITESPACE
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 — NORMALIZING TEXT WHITESPACE")
print("=" * 70)

df["text"] = (
    df["text"]
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
    .str.strip()
)


# ============================================================
# STEP 9 — CHECK CUSTOMER / AMAZONHELP COUNTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 — FINAL ROLE COUNTS")
print("=" * 70)

customer_count = (
    df["inbound"] == True
).sum()

amazon_count = (
    df["inbound"] == False
).sum()

print(
    f"\nCustomer messages: "
    f"{customer_count:,}"
)

print(
    f"AmazonHelp messages: "
    f"{amazon_count:,}"
)


# ============================================================
# STEP 10 — SAVE CLEAN DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 — SAVING CLEAN DATA")
print("=" * 70)

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

final_count = len(df)

removed_count = (
    initial_count - final_count
)

retention_rate = (
    final_count / initial_count
    if initial_count > 0
    else 0
)

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(
    f"\nInitial tweets: "
    f"{initial_count:,}"
)

print(
    f"Final tweets: "
    f"{final_count:,}"
)

print(
    f"Removed: "
    f"{removed_count:,}"
)

print(
    f"Retention rate: "
    f"{retention_rate:.2%}"
)

print(
    f"\nSaved to:"
    f"\n{OUTPUT_PATH}"
)