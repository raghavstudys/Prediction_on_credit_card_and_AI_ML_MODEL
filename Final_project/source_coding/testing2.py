import nltk
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
import os

vader_lexicon_path = os.path.expanduser("/Users/shanthakumark/Downloads/vader_lexicon.txt")

@st.cache_resource
def load_sentiment_analyzer():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.data.load(vader_lexicon_path)  # Loading from local file
    return SentimentIntensityAnalyzer()

# Analyze the sentiment of the text
def analyze_sentiment(text):
    sia = load_sentiment_analyzer()
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    st.title("Mobile Review Sentiment Analysis")

    # Input box for user review
    user_input = st.text_input("Enter your mobile review:")

    if user_input:
        sentiment = analyze_sentiment(user_input)
        st.write(f"The sentiment of the review is: **{sentiment}**")

if __name__ == '__main__':
    main()
