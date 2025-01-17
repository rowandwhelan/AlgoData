import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import os

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Configuration
API_KEY = os.getenv('NEWSAPI_KEY')  # Set as environment variable in GitHub Secrets
SYMBOLS = ['SPY', 'QQQ', 'IWM']     # Add your target symbols here
FROM_DATE = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')  # Yesterday
TO_DATE = datetime.utcnow().strftime('%Y-%m-%d')                        # Today
CSV_FILE_PATH = 'news_sentiment.csv'                                    # Path to your CSV file

def fetch_news(symbol, api_key, from_date, to_date, page_size=100):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': symbol,                  # Query keyword
        'from': from_date,            # Start date (YYYY-MM-DD)
        'to': to_date,                # End date (YYYY-MM-DD)
        'language': 'en',             # Language
        'sortBy': 'relevancy',        # Sorting
        'pageSize': page_size,        # Number of articles per request (max 100)
        'apiKey': api_key             # Your API key
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data['status'] == 'ok':
        return data['articles']
    else:
        print(f"Error fetching news for {symbol}: {data.get('message', 'Unknown error')}")
        return []

def analyze_sentiment(text):
    if not text:
        return 0.0  # Neutral if no content
    vs = analyzer.polarity_scores(text)
    return vs['compound']  # Compound score ranges from -1 (extremely negative) to +1 (extremely positive)

def aggregate_daily_sentiment(articles):
    sentiment_per_date = defaultdict(list)
    for article in articles:
        published_at = article.get('publishedAt', '')[:10]  # 'YYYY-MM-DD'
        if not published_at:
            continue  # Skip if no publication date
        try:
            date = datetime.strptime(published_at, "%Y-%m-%d").date()
        except ValueError:
            continue  # Skip if date format is incorrect
        content = article.get('content') or article.get('description') or ""
        sentiment = analyze_sentiment(content)
        sentiment_per_date[date].append(sentiment)
    # Calculate average sentiment per day
    avg_sentiment = {date: round(sum(scores)/len(scores), 2) for date, scores in sentiment_per_date.items() if scores}
    return avg_sentiment

def update_csv(symbol, sentiment_data, csv_path):
    # Read existing data
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
    except FileNotFoundError:
        # If the file does not exist, create a new DataFrame with headers
        df = pd.DataFrame(columns=['symbol', 'date', 'sentiment'])

    # Convert existing dates to set for quick lookup
    existing_dates = set(df[df['symbol'] == symbol]['date'].dt.date)

    # Prepare new data
    new_rows = []
    for date, sentiment in sentiment_data.items():
        if date not in existing_dates:
            new_rows.append({'symbol': symbol, 'date': date, 'sentiment': sentiment})

    # Append new data
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        # Save back to CSV
        df.to_csv(csv_path, index=False)
        print(f"Updated CSV with {len(new_rows)} new records for {symbol}.")
    else:
        print(f"No new records to add for {symbol}.")

def main():
    for symbol in SYMBOLS:
        print(f"Processing symbol: {symbol}")
        articles = fetch_news(symbol, API_KEY, FROM_DATE, TO_DATE)
        print(f"Fetched {len(articles)} articles for {symbol}.")
        sentiment_data = aggregate_daily_sentiment(articles)
        print(f"Aggregated sentiment for {symbol}: {sentiment_data}")
        update_csv(symbol, sentiment_data, CSV_FILE_PATH)

if __name__ == "__main__":
    main()
