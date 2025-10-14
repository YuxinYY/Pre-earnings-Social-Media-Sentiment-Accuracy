'''
PART I
this part contains helper functions that makes sure text data can be paired with stocks
'''

import re
import pandas as pd

def create_smart_stock_patterns(df_stocks):
    SKIP_TICKERS = {
        'A',     # Too common as article
        'I',     # Too common as pronoun  
        'AM',    # Common word "am"
        'AN',    # Common word "an"
        'AS',    # Common word "as"
        'AT',    # Common word "at"
        'BE',    # Common word "be" 
        'BY',    # Common word "by"
        'DO',    # Common word "do"
        'GO',    # Common word "go"
        'HE',    # Common word "he"
        'IF',    # Common word "if"
        'IN',    # Common word "in"
        'IS',    # Common word "is"
        'IT',    # Common word "it"
        'MY',    # Common word "my"
        'NO',    # Common word "no"
        'OF',    # Common word "of"
        'ON',    # Common word "on"
        'OR',    # Common word "or"
        'SO',    # Common word "so"
        'TO',    # Common word "to"
        'UP',    # Common word "up"
        'US',    # Common word "us"
        'WE',    # Common word "we"
        # Add more if needed
    }
    
    # Minimum length for ticker matching
    MIN_TICKER_LENGTH = 2
    
    stock_patterns = {}
    
    for _, row in df_stocks.iterrows():
        ticker = row['ticker'].upper()
        name = row['name']
        title = row['title']
        
        patterns = {
            'safe_tickers': [],      # Tickers safe to match
            'company_names': [],     # Company name variations
            'require_context': []    # Tickers that need context (like "$A")
        }
        
        # Handle ticker matching
        if ticker in SKIP_TICKERS:
            # For problematic tickers, only match with $ prefix or clear stock context
            patterns['require_context'].append(ticker)
        elif len(ticker) >= MIN_TICKER_LENGTH:
            patterns['safe_tickers'].append(ticker)
        
        # Add company names (cleaned)
        clean_name = re.sub(r'\b(inc\.?|corp\.?|corporation|company|co\.?|ltd\.?)\b', '', name, flags=re.IGNORECASE).strip()
        clean_title = re.sub(r'\b(inc\.?|corp\.?|corporation|company|co\.?|ltd\.?)\b', '', title, flags=re.IGNORECASE).strip()
        
        patterns['company_names'].extend([name, title, clean_name, clean_title])
        
        # Remove empty/short names
        patterns['company_names'] = [n for n in patterns['company_names'] if len(n.strip()) > 2]
        
        stock_patterns[ticker] = patterns
    
    return stock_patterns


def fast_stock_mentions(text, stock_patterns):
    text_upper = text.upper()
    text_lower = text.lower()
    mentions = []
    
    for ticker, patterns in stock_patterns.items():
        if re.search(r'\b' + re.escape(ticker) + r'\b', text_upper):
            mentions.append({
                'ticker': ticker,
                'match_type': 'exact_ticker',
                'confidence': 100
            })
            continue
        
        for pattern in patterns:
            if pattern.lower() in text_lower:
                mentions.append({
                    'ticker': ticker,
                    'match_type': 'exact_name', 
                    'confidence': 95
                })
                break
    
    return mentions

def analyze_stock_mentions_fast(df_social, stock_patterns, batch_size=10000): # batch size can be adjusted
    results = []
    total_rows = len(df_social)
    
    essential_fields = [ # these are columns to be preserved in the df_comments dataframe; submissions have title and selftext, while comments have body
    
        'id', 'title', 'selftext', 'body', 'author', 'created_utc', 'score', 
        'num_comments', 'url', 'subreddit', 'upvote_ratio'
    ]
    
    for i in range(0, total_rows, batch_size):
        batch = df_social.iloc[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1}")
        
        for idx, row in batch.iterrows():
            text = str(row.get('selftext', '')) + ' ' + str(row.get('title', '')) + ' '+ str(row.get('body', ''))
            mentions = fast_stock_mentions(text, stock_patterns)  # Use fast version
            
            # Create base record with all essential fields
            base_record = {}
            for field in essential_fields:
                base_record[field] = row.get(field, None)
            
            for mention in mentions:
                # Create a complete record for each mention
                record = base_record.copy()  # Copy all the essential fields
                
                # Add stock mention specific fields
                record.update({
                    'ticker': mention['ticker'],
                    'match_type': mention['match_type'],
                    'confidence': mention['confidence']
                })
                
                results.append(record)
    
    return pd.DataFrame(results)

'''
PART II
This part contains helper functions in the training process
'''
def create_sequences(row, df_daily, lookback=60): #you may change the look-back period based on needs
    ticker = row['ticker']
    earnings_date = pd.to_datetime(row['reportedDate'])
    
    # Get date range
    start_date = earnings_date - pd.Timedelta(days=lookback)
    end_date = earnings_date - pd.Timedelta(days=1)  # the day before earnings report #note that this is the day when the report comes out, not the date of the fiscal end
    
    mask = (
        (df_daily['ticker'] == ticker) &
        (df_daily['date'] >= start_date) &
        (df_daily['date'] <= end_date)
    )
    sequence_data = df_daily[mask].sort_values('date')
    
    # Handle missing days (days with no posts)
    date_range = pd.date_range(start_date, end_date, freq='D')
    sequence_data = sequence_data.set_index('date').reindex(date_range)
    sequence_data['combined_text'] = sequence_data['combined_text'].fillna('')  # empty string for no posts
    
    return sequence_data['combined_text'].tolist()
