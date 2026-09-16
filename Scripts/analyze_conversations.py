import pandas as pd
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/amazon_support.csv"

BRAND = "AmazonHelp"


# ============================================================
# STEP 1 — LOAD EXTRACTED DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — LOADING AMAZONHELP DATA")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH,
    dtype={
        "tweet_id": "string",
        "author_id": "string",
        "inbound": "boolean",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string",
    },
)

print(
    f"\nLoaded {len(df):,} tweets."
)


# ============================================================
# STEP 2 — BASIC DATASET STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — BASIC STATISTICS")
print("=" * 70)

total_tweets = len(df)

customer_tweets = (
    df["inbound"] == True
).sum()

brand_tweets = (
    df["inbound"] == False
).sum()

unique_authors = (
    df["author_id"]
    .dropna()
    .nunique()
)

unique_customers = (
    df.loc[
        df["inbound"] == True,
        "author_id"
    ]
    .dropna()
    .nunique()
)

print(
    f"\nTotal tweets:       {total_tweets:,}"
)

print(
    f"Customer tweets:    {customer_tweets:,}"
)

print(
    f"AmazonHelp tweets:  {brand_tweets:,}"
)

print(
    f"Unique authors:     {unique_authors:,}"
)

print(
    f"Unique customers:   {unique_customers:,}"
)


# ============================================================
# STEP 3 — BUILD TWEET LOOKUP
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — BUILDING TWEET RELATIONSHIPS")
print("=" * 70)

tweet_lookup = {}

for row in df.itertuples(index=False):

    tweet_id = row.tweet_id

    if pd.isna(tweet_id):
        continue

    tweet_lookup[str(tweet_id)] = {
        "author_id": row.author_id,
        "inbound": row.inbound,
        "text": row.text,
        "created_at": row.created_at,
        "parent_id": row.in_response_to_tweet_id,
    }


print(
    f"Tweet relationships built: "
    f"{len(tweet_lookup):,}"
)


# ============================================================
# STEP 4 — BUILD PARENT → CHILD RELATIONSHIPS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — BUILDING CONVERSATION THREADS")
print("=" * 70)

children = defaultdict(list)

for tweet_id, tweet in tweet_lookup.items():

    parent_id = tweet["parent_id"]

    if pd.isna(parent_id):
        continue

    parent_id = str(parent_id)

    if parent_id in tweet_lookup:

        children[parent_id].append(
            tweet_id
        )


print(
    f"Parent tweets with replies: "
    f"{len(children):,}"
)


# ============================================================
# STEP 5 — FIND CONVERSATION ROOT
# ============================================================

def find_root(tweet_id):
    """
    Follow parent relationships until we reach
    the beginning of the thread.
    """

    current = tweet_id

    visited = set()

    while True:

        if current in visited:
            break

        visited.add(current)

        tweet = tweet_lookup.get(current)

        if tweet is None:
            break

        parent_id = tweet["parent_id"]

        if pd.isna(parent_id):
            break

        parent_id = str(parent_id)

        if parent_id not in tweet_lookup:
            break

        current = parent_id

    return current


# ============================================================
# STEP 6 — ASSIGN CONVERSATION IDs
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — ASSIGNING CONVERSATION IDs")
print("=" * 70)

tweet_to_conversation = {}

for tweet_id in tweet_lookup:

    root = find_root(tweet_id)

    tweet_to_conversation[
        tweet_id
    ] = root


print(
    f"Conversation IDs assigned to "
    f"{len(tweet_to_conversation):,} tweets."
)


# ============================================================
# STEP 7 — ANALYZE CONVERSATIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — ANALYZING CONVERSATIONS")
print("=" * 70)

conversation_tweets = defaultdict(list)

for tweet_id, conversation_id in (
    tweet_to_conversation.items()
):

    conversation_tweets[
        conversation_id
    ].append(tweet_id)


# ------------------------------------------------------------
# Only conversations involving AmazonHelp
# ------------------------------------------------------------

amazon_conversations = {}

for conversation_id, tweet_ids in (
    conversation_tweets.items()
):

    contains_amazon = False

    for tweet_id in tweet_ids:

        tweet = tweet_lookup[tweet_id]

        if tweet["author_id"] == BRAND:

            contains_amazon = True
            break

    if contains_amazon:

        amazon_conversations[
            conversation_id
        ] = tweet_ids


print(
    f"\nAmazonHelp conversation threads: "
    f"{len(amazon_conversations):,}"
)


# ============================================================
# STEP 8 — KEEP ONLY GENUINE CUSTOMER + AMAZONHELP THREADS
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 — FILTERING GENUINE SUPPORT CONVERSATIONS")
print("=" * 70)

genuine_conversations = {}

for conversation_id, tweet_ids in (
    amazon_conversations.items()
):

    has_customer = False
    has_amazon = False

    for tweet_id in tweet_ids:

        tweet = tweet_lookup[tweet_id]

        if tweet["inbound"] == True:
            has_customer = True

        elif tweet["author_id"] == BRAND:
            has_amazon = True

    if has_customer and has_amazon:

        genuine_conversations[
            conversation_id
        ] = tweet_ids


print(
    f"\nGenuine customer + AmazonHelp "
    f"conversations: "
    f"{len(genuine_conversations):,}"
)


# ============================================================
# STEP 8 — CONVERSATION LENGTH
# ============================================================

conversation_lengths = []

for conversation_id, tweet_ids in (
    genuine_conversations.items()
):

    conversation_lengths.append(
        len(tweet_ids)
    )


lengths = pd.Series(
    conversation_lengths
)


# ============================================================
# STEP 9 — ONE-TURN VS MULTI-TURN
# ============================================================

one_turn = (
    lengths == 1
).sum()

multi_turn = (
    lengths >= 2
).sum()

three_plus = (
    lengths >= 3
).sum()

five_plus = (
    lengths >= 5
).sum()


# ============================================================
# STEP 10 — CUSTOMER / AMAZONHELP MESSAGE COUNTS
# ============================================================

total_customer_messages = 0
total_amazon_messages = 0

for conversation_id, tweet_ids in (
    genuine_conversations.items()
):

    for tweet_id in tweet_ids:

        tweet = tweet_lookup[tweet_id]

        if tweet["inbound"] == True:

            total_customer_messages += 1

        elif tweet["author_id"] == BRAND:

            total_amazon_messages += 1


# ============================================================
# STEP 11 — PRINT GENUINE CONVERSATION RESULTS
# ============================================================

print("\n" + "=" * 80)
print("GENUINE AMAZONHELP CONVERSATION ANALYSIS")
print("=" * 80)

total_conversations = len(
    genuine_conversations
)

print(
    f"\nGenuine conversations: "
    f"{total_conversations:,}"
)

print(
    f"Customer messages: "
    f"{total_customer_messages:,}"
)

print(
    f"AmazonHelp messages: "
    f"{total_amazon_messages:,}"
)

print(
    f"\nOne-turn conversations: "
    f"{one_turn:,}"
)

print(
    f"Multi-turn conversations: "
    f"{multi_turn:,}"
)

print(
    f"3+ message conversations: "
    f"{three_plus:,}"
)

print(
    f"5+ message conversations: "
    f"{five_plus:,}"
)


# ============================================================
# STEP 12 — PERCENTAGES
# ============================================================

print("\n" + "=" * 80)
print("GENUINE CONVERSATION PERCENTAGES")
print("=" * 80)

if total_conversations > 0:

    print(
        f"\nOne-turn: "
        f"{one_turn / total_conversations:.2%}"
    )

    print(
        f"Multi-turn: "
        f"{multi_turn / total_conversations:.2%}"
    )

    print(
        f"3+ messages: "
        f"{three_plus / total_conversations:.2%}"
    )

    print(
        f"5+ messages: "
        f"{five_plus / total_conversations:.2%}"
    )


# ============================================================
# STEP 13 — LENGTH STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("GENUINE CONVERSATION LENGTH")
print("=" * 80)

if not lengths.empty:

    print(
        f"\nAverage: "
        f"{lengths.mean():.2f} tweets"
    )

    print(
        f"Median: "
        f"{lengths.median():.0f} tweets"
    )

    print(
        f"Minimum: "
        f"{lengths.min()} tweet"
    )

    print(
        f"Maximum: "
        f"{lengths.max()} tweets"
    )

    print(
        f"75th percentile: "
        f"{lengths.quantile(0.75):.0f} tweets"
    )

    print(
        f"90th percentile: "
        f"{lengths.quantile(0.90):.0f} tweets"
    )