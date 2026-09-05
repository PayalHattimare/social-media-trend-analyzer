import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Social Media Trend Analyzer",
    page_icon="📱",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📱 Social Media Trend Analyzer")

st.write(
    "NLP-based analysis of social media posts, "
    "sentiment, hashtags, topics and engagement."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "social_media_trend_analyzer_dataset.csv"
    )

    return df


df = load_data()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["clean_text"] = df["text"].apply(
    clean_text
)


# ============================================================
# ENGAGEMENT
# ============================================================

df["engagement"] = (
    df["likes"]
    + df["shares"]
    + df["comments"]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Filters")

sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment",
    options=df["sentiment"].unique(),
    default=df["sentiment"].unique()
)

topic_filter = st.sidebar.multiselect(
    "Select Topic",
    options=df["topic"].unique(),
    default=df["topic"].unique()
)


filtered_df = df[
    (df["sentiment"].isin(sentiment_filter))
    &
    (df["topic"].isin(topic_filter))
]


# ============================================================
# KPI SECTION
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Posts",
        len(filtered_df)
    )


with col2:

    st.metric(
        "Total Likes",
        int(filtered_df["likes"].sum())
    )


with col3:

    st.metric(
        "Total Shares",
        int(filtered_df["shares"].sum())
    )


with col4:

    st.metric(
        "Total Engagement",
        int(filtered_df["engagement"].sum())
    )


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

st.header("😊 Sentiment Analysis")

col1, col2 = st.columns(2)


with col1:

    sentiment_counts = (
        filtered_df["sentiment"]
        .value_counts()
    )

    st.bar_chart(
        sentiment_counts
    )


with col2:

    st.write(
        "Sentiment Distribution"
    )

    st.dataframe(
        sentiment_counts
    )


# ============================================================
# TOPIC ANALYSIS
# ============================================================

st.header("📚 Topic Analysis")

topic_counts = (
    filtered_df["topic"]
    .value_counts()
)

st.bar_chart(
    topic_counts
)


# ============================================================
# HASHTAG ANALYSIS
# ============================================================

st.header("🔥 Trending Hashtags")

hashtags = []

for value in filtered_df["hashtags"]:

    tags = re.findall(
        r"#\w+",
        str(value)
    )

    hashtags.extend(tags)


hashtag_counts = Counter(
    hashtags
)

top_hashtags = pd.DataFrame(
    hashtag_counts.most_common(10),
    columns=["Hashtag", "Frequency"]
)

st.dataframe(
    top_hashtags,
    use_container_width=True
)


# ============================================================
# ENGAGEMENT ANALYSIS
# ============================================================

st.header("📊 Engagement Analysis")

engagement_topic = (
    filtered_df
    .groupby("topic")["engagement"]
    .mean()
    .sort_values(
        ascending=False
    )
)

st.bar_chart(
    engagement_topic
)


# ============================================================
# TOP POSTS
# ============================================================

st.header("🏆 Most Engaging Posts")

top_posts = (
    filtered_df
    .sort_values(
        "engagement",
        ascending=False
    )
    .head(10)
)


st.dataframe(
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
    ],
    use_container_width=True
)


# ============================================================
# TF-IDF ANALYSIS
# ============================================================

st.header("🔤 Important NLP Keywords")

vectorizer = TfidfVectorizer(
    max_features=50,
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    filtered_df["clean_text"]
)

scores = tfidf_matrix.sum(
    axis=0
).A1

words = vectorizer.get_feature_names_out()

keyword_df = pd.DataFrame(
    {
        "Keyword": words,
        "TF-IDF Score": scores
    }
)

keyword_df = keyword_df.sort_values(
    "TF-IDF Score",
    ascending=False
)

st.dataframe(
    keyword_df.head(20),
    use_container_width=True
)


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

st.header("🤖 Predict Sentiment")

texts = df["clean_text"]

labels = df["sentiment"]

vectorizer_model = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)

X = vectorizer_model.fit_transform(
    texts
)

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X,
    labels
)


user_text = st.text_area(
    "Enter a social media post:"
)


if st.button("Predict Sentiment"):

    if user_text.strip():

        cleaned = clean_text(
            user_text
        )

        transformed = vectorizer_model.transform(
            [cleaned]
        )

        prediction = model.predict(
            transformed
        )[0]

        confidence = (
            model.predict_proba(
                transformed
            ).max()
        )

        st.success(
            f"Predicted Sentiment: {prediction}"
        )

        st.info(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

    else:

        st.warning(
            "Please enter some text."
        )


# ============================================================
# DATASET
# ============================================================

st.header("📄 Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True
)