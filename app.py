import streamlit as st
import pandas as pd
import re
import os
import sys
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

# Custom CSS for YouTube-like theme
def youtube_theme():
    st.markdown("""
    <style>
    /* YouTube-inspired color palette */
    .stApp {
        background-color: #f9f9f9;
        color: #030303;
    }
        /* Top left YouTube icon */
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
        border: 8px solid red;  /* Thick red border */
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        background-color: white;
    }
    /* Small icons throughout the page */
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
    /* Title styling */
    .title {
        font-family: 'YouTube Sans', 'Roboto', sans-serif;
        color: #FF0000;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    /* Button styling */
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
    
    /* DataTable styling */
    .dataframe {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
    }
    
    /* Card-like containers */
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
    """
    Extract YouTube video ID from various URL formats
    """
    # List of possible YouTube URL patterns
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&\s]+)',  # Standard URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^&\s]+)',   # Embed URL
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^&\s]+)',             # Shortened URL
    ]
    
    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # If no match found, return None
    return None

def detect_language(text):
    """
    Basic language detection using character ranges
    """
    # Character ranges for different languages
    language_ranges = {
        'Devanagari (Hindi/Marathi)': (0x0900, 0x097F),
        'English': (0x0000, 0x007F),
        'Extended Latin': (0x0080, 0x00FF)
    }
    
    def is_in_range(char, start, end):
        return start <= ord(char) <= end
    
    # Check for Devanagari characters
    devanagari_chars = [char for char in text if is_in_range(char, 0x0900, 0x097F)]
    
    if devanagari_chars and len(devanagari_chars) / len(text) > 0.3:
        # Determine between Hindi and Marathi (simplified)
        return 'Hindi/Marathi'
    
    return 'English'

def clean_multilingual_text(text, language):
    """
    Clean text based on the detected language
    """
    # Remove URLs, special characters
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Language-specific cleaning
    if language == 'Hindi/Marathi':
        # Remove punctuation and special characters specific to Devanagari script
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    else:
        # For other languages, standard cleaning
        text = re.sub(r"[^\w\s]", "", text)
    
    # Lowercase
    text = text.lower()
    
    return text

def get_multilingual_sentiment(text, language):
    """
    Get sentiment for multilingual text
    """
    try:
        # Use TextBlob for sentiment analysis
        analysis = TextBlob(text)
        
        # Determine sentiment based on polarity
        if analysis.sentiment.polarity > 0:
            return "Positive"
        elif analysis.sentiment.polarity < 0:
            return "Negative"
        else:
            return "Neutral"
    
    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
        return "Neutral"

def get_youtube_comments(video_id, max_results=500):
    """
    Fetch YouTube comments using pagination.
    """
    API_KEY = os.getenv('YOUTUBE_API_KEY')
    if not API_KEY:
        raise ValueError("YouTube API key is not set. Please check your .env file.")
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    comments = []
    next_page_token = None
    
    try:
        while len(comments) < max_results:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),  # Request in chunks
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
                
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break  # Stop if no more pages available
                
        return comments

    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []


def main():
    # Apply YouTube theme
    youtube_theme()
    
    # Custom title with YouTube icon
    st.markdown('<h1 class="title">🎥 YouTube Comment Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    Dive deep into the sentiment of YouTube comments across multiple languages! 
    Paste a full YouTube video URL or just the video ID.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input for YouTube video URL
    video_url = st.text_input("🔗 Enter YouTube Video URL or Video ID:", "")
    
    if st.button("Analyze Comments") and video_url:
        # Extract video ID
        video_id = extract_video_id(video_url)
        
        # If extraction fails, assume the input might be a direct video ID
        if not video_id:
            video_id = video_url.strip()
        
        # Validate video ID
        if not video_id:
            st.error("Invalid YouTube URL or Video ID. Please check and try again.")
            return
        
        # Fetch comments
        with st.spinner('Fetching and analyzing comments...'):
            comments = get_youtube_comments(video_id)
        
        if not comments:
            st.warning("No comments found or error in fetching comments.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(comments, columns=["Comment"])
        
        # Detect languages and clean comments
        df['Language'] = df['Comment'].apply(detect_language)
        df['Cleaned_Comment'] = df.apply(lambda row: clean_multilingual_text(row['Comment'], row['Language']), axis=1)
        
        # Perform sentiment analysis
        df['Sentiment'] = df.apply(lambda row: get_multilingual_sentiment(row['Comment'], row['Language']), axis=1)
        
        # Display results in a card-like container
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Comment Analysis")
        st.dataframe(df[['Comment', 'Language', 'Sentiment']])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Language distribution
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🌍 Language Distribution")
        lang_counts = df['Language'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        lang_counts.plot(kind='bar', ax=ax, color='#FF0000')
        plt.title('Languages in Comments', color='#FF0000')
        plt.xlabel('Language', color='#333')
        plt.ylabel('Count', color='#333')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sentiment distribution
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("😊 Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sentiment_palette = {'Positive': '#48c774', 'Negative': '#f14668', 'Neutral': '#3273dc'}
        sns.countplot(x='Sentiment', data=df, palette=sentiment_palette, ax=ax)
        plt.title('Comment Sentiment Analysis', color='#FF0000')
        plt.xlabel('Sentiment', color='#333')
        plt.ylabel('Count', color='#333')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Word cloud
        # Display Top 5 Positive and Negative Comments
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏆 Top 5 Positive Comments")
        
        # Sort comments by sentiment polarity
        df['Polarity'] = df['Cleaned_Comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
        
        # Select Top 5 Positive Comments
        top_positive = df.sort_values(by='Polarity', ascending=False).head(5)
        
        # Show comments
        for idx, row in top_positive.iterrows():
            st.success(f"**{idx+1}.** {row['Comment']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("⚠️ Top 5 Negative Comments")
        
        # Select Top 5 Negative Comments
        top_negative = df.sort_values(by='Polarity', ascending=True).head(5)
        
        # Show comments
        for idx, row in top_negative.iterrows():
            st.error(f"**{idx+1}.** {row['Comment']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        # Display YouTube logo at the top left
        st.markdown(
            """
            <div class="youtube-header">
                <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg">
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display small icons (Like, Dislike, Comments, Report)
        st.markdown(
            """
            <div class="icon-row">
                <img src="https://cdn-icons-png.flaticon.com/512/1077/1077035.png" title="Like">
                <img src="https://cdn-icons-png.flaticon.com/512/126/126473.png" title="Dislike">
                <img src="https://cdn-icons-png.flaticon.com/512/1380/1380338.png" title="Comments">
                <img src="https://cdn-icons-png.flaticon.com/512/1828/1828843.png" title="Report">
            </div>
            """,
            unsafe_allow_html=True
        )



if __name__ == "__main__":
    main()

