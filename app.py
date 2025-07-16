import streamlit as st
import pandas as pd
import re
import os
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()

# Ensure NLTK downloads
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

def youtube_theme():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f9f9f9;
        color: #030303;
    }
    .youtube-header {
        display: flex;
        align-items: center;
        gap: 10px;
        position: absolute;
        top: 10px;
        left: 15px;
    }
    .youtube-header img {
        width: 50px;
        height: auto;
    }
    .content-container {
        border: 8px solid red;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        background-color: white;
    }
    .icon-row {
        display: flex;
        gap: 15px;
        justify-content: center;
        margin: 20px 0;
    }
    .icon-row img {
        width: 30px;
        height: auto;
        cursor: pointer;
        transition: transform 0.2s ease-in-out;
    }
    .icon-row img:hover {
        transform: scale(1.2);
    }
    .title {
        font-family: 'YouTube Sans', 'Roboto', sans-serif;
        color: #FF0000;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stButton > button {
        background-color: #FF0000;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 2px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #CC0000;
    }
    .dataframe {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
    }
    .card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def extract_video_id(url):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&\s]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^&\s]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^&\s]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def detect_language(text):
    devanagari_chars = [char for char in text if 0x0900 <= ord(char) <= 0x097F]
    if devanagari_chars and len(devanagari_chars) / len(text) > 0.3:
        return 'Hindi/Marathi'
    return 'English'

def clean_multilingual_text(text, language):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    if language == 'Hindi/Marathi':
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    else:
        text = re.sub(r"[^\w\s]", "", text)
    return text.lower()

def get_multilingual_sentiment(text, language):
    try:
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return "Positive"
        elif analysis.sentiment.polarity < 0:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
        return "Neutral"

def get_youtube_comments(video_id, max_results=300):
    API_KEY = os.getenv('YOUTUBE_API_KEY')
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    next_page_token = None
    try:
        while len(comments) < max_results:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                pageToken=next_page_token
            )
            response = request.execute()
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        return comments
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []

def main():
    youtube_theme()
    st.markdown('<h1 class="title">🎥 YouTube Comment Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<div class="card">Dive deep into the sentiment of YouTube comments across multiple languages!</div>', unsafe_allow_html=True)
    
    video_url = st.text_input("🔗 Enter YouTube Video URL or Video ID:", "")
    
    if st.button("Analyze Comments") and video_url:
        video_id = extract_video_id(video_url) or video_url.strip()
        if not video_id:
            st.error("Invalid YouTube URL or Video ID.")
            return

        with st.spinner('Fetching and analyzing comments...'):
            comments = get_youtube_comments(video_id)
        
        if not comments:
            st.warning("No comments found.")
            return

        df = pd.DataFrame(comments, columns=["Comment"])
        df['Language'] = df['Comment'].apply(detect_language)
        df['Cleaned_Comment'] = df.apply(lambda row: clean_multilingual_text(row['Comment'], row['Language']), axis=1)
        df['Sentiment'] = df.apply(lambda row: get_multilingual_sentiment(row['Cleaned_Comment'], row['Language']), axis=1)
        df['Polarity'] = df['Cleaned_Comment'].apply(lambda x: TextBlob(x).sentiment.polarity)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Comment Analysis")
        st.dataframe(df[['Comment', 'Language', 'Sentiment']])
        st.markdown('</div>', unsafe_allow_html=True)

        # Language distribution
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🌍 Language Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        df['Language'].value_counts().plot(kind='bar', ax=ax, color='#FF0000')
        plt.title('Languages in Comments', color='#FF0000')
        plt.xlabel('Language')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Sentiment distribution
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("😊 Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Sentiment', data=df, palette={'Positive': '#48c774', 'Negative': '#f14668', 'Neutral': '#3273dc'}, ax=ax)
        plt.title('Comment Sentiment Analysis', color='#FF0000')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Top 5 Positive Comments
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color:#218838; padding:8px 12px; border-radius:5px; color:white; font-size:20px; font-weight:bold; margin-bottom:10px;">
        🏆 Top 5 Positive Comments
        </div>
        """, unsafe_allow_html=True)
        top_positive = df.sort_values(by='Polarity', ascending=False).head(5)
        for idx, row in top_positive.iterrows():
            st.markdown(f"""
                <div style="background-color:#218838; padding:10px; border-radius:5px; color:white; margin-bottom:10px;">
                    <strong>{idx+1}.</strong> {row['Comment']}
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Top 5 Negative Comments
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color:#b02a37; padding:8px 12px; border-radius:5px; color:white; font-size:20px; font-weight:bold; margin-bottom:10px;">
         ⚠️ Top 5 Negative Comments
         </div>
        """, unsafe_allow_html=True)
        top_negative = df.sort_values(by='Polarity', ascending=True).head(5)
        for idx, row in top_negative.iterrows():
            st.markdown(f"""
                <div style="background-color:#b02a37; padding:10px; border-radius:5px; color:white; margin-bottom:10px;">
                    <strong>{idx+1}.</strong> {row['Comment']}
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # YouTube Logo & Icons
        st.markdown("""
        <div class="youtube-header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg">
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="icon-row">
            <img src="https://cdn-icons-png.flaticon.com/512/1077/1077035.png" title="Like">
            <img src="https://cdn-icons-png.flaticon.com/512/126/126473.png" title="Dislike">
            <img src="https://cdn-icons-png.flaticon.com/512/1380/1380338.png" title="Comments">
            <img src="https://cdn-icons-png.flaticon.com/512/1828/1828843.png" title="Report">
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
