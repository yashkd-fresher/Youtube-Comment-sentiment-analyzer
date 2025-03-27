# YouTube Comment Sentiment Analyzer

## Overview
A Streamlit application that analyzes YouTube video comments for sentiment and language distribution.

## Features
- Fetch YouTube video comments
- Multilingual sentiment analysis
- Language detection
- Visualization of comment sentiments
- Top positive and negative comments display

## Prerequisites
- Python 3.8+
- YouTube Data API Key

## Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/youtube-comment-sentiment.git
cd youtube-comment-sentiment
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
- Create a `.env` file in the project root
- Add your YouTube API key:
```
YOUTUBE_API_KEY=your_api_key_here
```

## Running the Application
```bash
streamlit run app.py
```

## Disclaimer
Ensure you comply with YouTube's Terms of Service and API usage policies.
