import pandas as pd
import re

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv("social_media_trend_analyzer_dataset.csv")

print("\n==============================================")
print("       SOCIAL MEDIA NLP TREND ANALYZER")
print("==============================================")

print("\nDataset loaded successfully!")

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))


# ==========================================================
# 2. DISPLAY DATA
# ==========================================================

print("\nFirst 5 records:")

print(df.head())


print("\nColumn names:")

print(df.columns.tolist())


# ==========================================================
# 3. CHECK MISSING VALUES
# ==========================================================

print("\nMissing values:")

print(df.isnull().sum())


# ==========================================================
# 4. HANDLE MISSING VALUES
# ==========================================================

df["text"] = df["text"].fillna("")
df["hashtags"] = df["hashtags"].fillna("")
df["sentiment"] = df["sentiment"].fillna("Unknown")
df["topic"] = df["topic"].fillna("Unknown")


# ==========================================================
# 5. TEXT PREPROCESSING
# ==========================================================

def clean_text(text):

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtag symbol
    text = re.sub(
        r"#",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Keep only alphabets and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Simple tokenization
    tokens = text.split()

    # Stop words
    stop_words = {
        "a", "an", "the", "and", "or",
        "is", "are", "was", "were",
        "to", "of", "in", "on", "for",
        "with", "this", "that", "it",
        "as", "at", "be", "by", "from",
        "has", "have", "had", "but",
        "not", "you", "your", "we", "our",
        "they", "their", "i", "me", "my",
        "he", "she", "his", "her"
    }

    # Remove stop words
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# Apply cleaning

df["clean_text"] = df["text"].apply(clean_text)


print("\n==============================================")
print("TEXT PREPROCESSING")
print("==============================================")


print("\nOriginal text:")

print(df["text"].head())


print("\nCleaned text:")

print(df["clean_text"].head())


# ==========================================================
# 6. WORD FREQUENCY
# ==========================================================

all_words = " ".join(
    df["clean_text"]
)

words = all_words.split()

word_frequency = Counter(words)


print("\n==============================================")
print("TOP 20 MOST COMMON WORDS")
print("==============================================")


for word, count in word_frequency.most_common(20):

    print(
        word,
        ":",
        count
    )


# ==========================================================
# 7. SENTIMENT ANALYSIS
# ==========================================================

print("\n==============================================")
print("SENTIMENT ANALYSIS")
print("==============================================")


sentiment_counts = df[
    "sentiment"
].value_counts()


print(
    sentiment_counts
)


# ==========================================================
# 8. TOPIC ANALYSIS
# ==========================================================

print("\n==============================================")
print("TOPIC ANALYSIS")
print("==============================================")


topic_counts = df[
    "topic"
].value_counts()


print(
    topic_counts
)


# ==========================================================
# 9. HASHTAG ANALYSIS
# ==========================================================

hashtags = []


for value in df["hashtags"]:

    tags = re.findall(
        r"#\w+",
        str(value)
    )

    hashtags.extend(tags)


hashtag_frequency = Counter(
    hashtags
)


print("\n==============================================")
print("TOP 20 HASHTAGS")
print("==============================================")


for hashtag, count in hashtag_frequency.most_common(20):

    print(
        hashtag,
        ":",
        count
    )


# ==========================================================
# 10. ENGAGEMENT ANALYSIS
# ==========================================================

df["engagement"] = (
    df["likes"]
    + df["shares"]
    + df["comments"]
)


print("\n==============================================")
print("ENGAGEMENT ANALYSIS")
print("==============================================")


print(
    df[
        [
            "likes",
            "shares",
            "comments",
            "engagement"
        ]
    ].describe()
)


# ==========================================================
# 11. TOP 10 MOST ENGAGING POSTS
# ==========================================================

top_posts = df.sort_values(
    by="engagement",
    ascending=False
).head(10)


print("\n==============================================")
print("TOP 10 MOST ENGAGING POSTS")
print("==============================================")


print(
    top_posts[
        [
            "text",
            "likes",
            "shares",
            "comments",
            "engagement",
            "sentiment",
            "topic"
        ]
    ].to_string(index=False)
)


# ==========================================================
# 12. TF-IDF
# ==========================================================

print("\n==============================================")
print("TF-IDF ANALYSIS")
print("==============================================")


tfidf_vectorizer = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2)
)


tfidf_matrix = tfidf_vectorizer.fit_transform(
    df["clean_text"]
)


print(
    "\nTF-IDF Matrix Shape:",
    tfidf_matrix.shape
)


# Get feature names

feature_names = (
    tfidf_vectorizer
    .get_feature_names_out()
)


# Calculate scores

tfidf_scores = (
    tfidf_matrix.sum(axis=0)
    .A1
)


tfidf_words = pd.DataFrame(
    {
        "word": feature_names,
        "score": tfidf_scores
    }
)


tfidf_words = tfidf_words.sort_values(
    by="score",
    ascending=False
)


print("\nTop 20 TF-IDF words:")


print(
    tfidf_words.head(20)
)


# ==========================================================
# 13. MACHINE LEARNING
# SENTIMENT CLASSIFICATION
# ==========================================================

print("\n==============================================")
print("MACHINE LEARNING")
print("SENTIMENT CLASSIFICATION")
print("==============================================")


# Remove Unknown sentiment

data = df[
    df["sentiment"] != "Unknown"
].copy()


X = data["clean_text"]

y = data["sentiment"]


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(
    "\nTraining records:",
    len(X_train)
)


print(
    "Testing records:",
    len(X_test)
)


# ==========================================================
# 14. TF-IDF FOR MACHINE LEARNING
# ==========================================================

model_vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)


X_train_tfidf = (
    model_vectorizer
    .fit_transform(X_train)
)


X_test_tfidf = (
    model_vectorizer
    .transform(X_test)
)


print(
    "\nTraining TF-IDF shape:",
    X_train_tfidf.shape
)


# ==========================================================
# 15. LOGISTIC REGRESSION MODEL
# ==========================================================

model = LogisticRegression(
    max_iter=1000
)


print("\nTraining model...")


model.fit(
    X_train_tfidf,
    y_train
)


print("Model training completed!")


# ==========================================================
# 16. PREDICTION
# ==========================================================

y_pred = model.predict(
    X_test_tfidf
)


# ==========================================================
# 17. MODEL ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n==============================================")
print("MODEL RESULTS")
print("==============================================")


print(
    "\nModel Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print("\nClassification Report:")


print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================================
# 18. CUSTOM SENTIMENT PREDICTION
# ==========================================================

def predict_sentiment(text):

    cleaned_text = clean_text(
        text
    )

    text_vector = (
        model_vectorizer
        .transform([cleaned_text])
    )

    prediction = model.predict(
        text_vector
    )[0]

    probability = (
        model.predict_proba(
            text_vector
        ).max()
    )

    return prediction, probability


print("\n==============================================")
print("CUSTOM SENTIMENT PREDICTION")
print("==============================================")


user_text = input(
    "\nEnter a social media post: "
)


prediction, probability = (
    predict_sentiment(
        user_text
    )
)


print(
    "\nPredicted Sentiment:",
    prediction
)


print(
    "Confidence:",
    round(
        probability * 100,
        2
    ),
    "%"
)


# ==========================================================
# 19. SAVE PROCESSED DATASET
# ==========================================================

df.to_csv(
    "processed_social_media_dataset.csv",
    index=False
)


print(
    "\nProcessed dataset saved as:"
)

print(
    "processed_social_media_dataset.csv"
)


# ==========================================================
# PROJECT COMPLETED
# ==========================================================

print("\n==============================================")
print("       NLP PROJECT COMPLETED SUCCESSFULLY")
print("==============================================")