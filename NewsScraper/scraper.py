# TODO: finish it with the finnhub api, make it work :)

import finnhub
import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('FINNHUB_API_KEY')
# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=API_KEY)

def fetch_news_with_checkpoint(tickers):
    """Fetches news with rate limiting and intermediate saving."""
    
    # Load existing data if it exists to resume progress
    if os.path.exists(NEWS_CSV):
        existing_df = pd.read_csv(NEWS_CSV)
        processed_tickers = existing_df['ticker'].unique().tolist()
    else:
        existing_df = pd.DataFrame()
        processed_tickers = []

    print(f"--- Starting News Scrape ({len(processed_tickers)}/{len(tickers)} completed) ---")

    for ticker in tickers:
        if ticker in processed_tickers:
            print(f"Skipping {ticker} (Already processed)")
            continue
            
        print(f"Processing {ticker}...")
        all_ticker_news = []
        
        # Finnhub works best with small date chunks (weekly) to avoid missing data
        current_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
        final_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
        
        while current_dt < final_dt:
            from_date = current_dt.strftime('%Y-%m-%d')
            to_date = (current_dt + timedelta(days=7)).strftime('%Y-%m-%d')
            
            try:
                # Fetch news for the 7-day window
                # Rate limit: 60 calls/min (we wait ~1.1s to be safe)
                news = finnhub_client.company_news(ticker, _from=from_date, to=to_date)
                
                for item in news:
                    all_ticker_news.append({
                        'ticker': ticker,
                        'date': datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d %H:%M:%S'),
                        'headline': item['headline'],
                        'summary': item['summary'],
                        'source': item['source'],
                        'url': item['url']
                    })
                
                time.sleep(1.1) # Respect Finnhub rate limit
                
            except Exception as e:
                print(f"Error fetching {ticker} on {from_date}: {e}")
                break # Move to next ticker or retry later
            
            current_dt += timedelta(days=8) # Move to next week
            
        # Append this ticker's data to the CSV immediately
        if all_ticker_news:
            ticker_df = pd.DataFrame(all_ticker_news)
            ticker_df.to_csv(NEWS_CSV, mode='a', header=not os.path.exists(NEWS_CSV), index=False)
            print(f"Saved {len(all_ticker_news)} articles for {ticker}.")
        else:
            print(f"No news found for {ticker} (Likely outside API date range).")
            # Log the skip so we don't try again
            pd.DataFrame([{'ticker': ticker, 'headline': 'NO_DATA'}]).to_csv(NEWS_CSV, mode='a', header=not os.path.exists(NEWS_CSV), index=False)

# --- EXECUTION ---
if __name__ == "__main__":
    sp100 = get_sp100_tickers()
    fetch_news_with_checkpoint(sp100)