'''
PART I
this part contains helper functions that makes sure text data can be paired with stocks
'''

import re
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from gliner import GLiNER
from flashtext import KeywordProcessor

def generate_full_text(row):
        ticker = str(row['ticker'])
        title = str(row['title']).strip()
        
        suffix_pattern = r'[,.\s]+(Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|Co\.?|Company|PLC|L\.P\.?|LLC|S\.A\.?|S\.A\.B\.?|de C\.V\.?|N\.V\.?|A\.?G\.?|S\.E\.?|S\.p\.A\.?|Group|Holdings?|Trust|Fund|ETF|REIT|/DE/?|/TX/?|/NY/?|/MD/?|/VA/?|/NV/?|/TN/?|/OH/?|/CN/?|/FI/?|/PA/?|/MA/?|/CA/?|& Co\.?|/NEW/|/OLD/).*$'
        
        alias = re.sub(suffix_pattern, '', title, flags=re.IGNORECASE).strip()
        alias = alias.strip('.,/- ')
        
        if not alias:
            alias = title
        return f"{title}, ticker: {ticker}, {alias}"


def build_cleaned_flashtext_processor(df_stocks, blacklist):
    keyword_processor = KeywordProcessor(case_sensitive=False)

    # white list
    valid_tickers = set(df_stocks['ticker'].str.upper().dropna().tolist())

    filtered_count = 0
    for _, row in df_stocks.iterrows():
        ticker = str(row['ticker']).upper()
        # title = str(row['title']) # Original title, less robust for aliases
        full_text = str(row['full_text']) # Use full_text which includes title and alias

        if ticker in blacklist:
            filtered_count += 1
            continue 

        keyword_processor.add_keyword(ticker, ticker)

        full_text_parts = [part.strip() for part in full_text.split(',') if part.strip()]

        if len(full_text_parts) > 0: 
            keyword_processor.add_keyword(full_text_parts[0], ticker)
        if len(full_text_parts) > 2: 
            keyword_processor.add_keyword(full_text_parts[2], ticker)

    return keyword_processor, valid_tickers


def local_process_comment_no_llm_or_embedding(keyword_processor, model, comment, names, comment_id, black_list, valid_tickers):
    hit_tickers = set()

    direct_hits = keyword_processor.extract_keywords(comment)
    hit_tickers.update(direct_hits)

    labels = ["company", "stock", "commercial organization"]
    entities = model.predict_entities(str(comment), labels, threshold=0.3)

    for entity in entities:
        text_span = entity['text']
        normalized_text_span = text_span.strip().upper()
        
        if normalized_text_span not in hit_tickers and normalized_text_span not in black_list:
            if normalized_text_span in names:
                matched_ticker = names.get(normalized_text_span)
                if matched_ticker and matched_ticker in valid_tickers: # Validate against original tickers
                    hit_tickers.add(matched_ticker)

    return list(hit_tickers)


'''
PART II
This part contains helper functions in the sequence embedding process
'''

#new sequence function:
def create_sequences(row, mentions, lookback=8): #you may change the look-back period based on needs
    ticker = row['ticker']
    returns_date = row['date']
    
    # Get date range
    start_date = returns_date - pd.Timedelta(days=lookback)
    end_date = returns_date - pd.Timedelta(days=1)  # the day before earnings report #note that this is the day when the report comes out, not the date of the fiscal end
    full_date_range = pd.date_range(start=start_date, end=end_date)
    
    mask = (
        (mentions['ticker'] == ticker) &
        (mentions['date'] >= start_date) &
        (mentions['date'] <= end_date)
    )
    daily_data = mentions[mask].sort_values('date')
    daily_groups = daily_data.groupby('date')['source_text'].apply(list)
    daily_groups = daily_groups.reindex(full_date_range, fill_value=[])

    #structure: [ ['Day1_Post1', 'Day1_Post2'], ['Day2_Post1'], [], ... ]
    return daily_groups.tolist()


# def create_sequences(row, df_daily, lookback=60): #you may change the look-back period based on needs
#     ticker = row['ticker']
#     earnings_date = pd.to_datetime(row['reportedDate'])
    
#     # Get date range
#     start_date = earnings_date - pd.Timedelta(days=lookback)
#     end_date = earnings_date - pd.Timedelta(days=1)  # the day before earnings report #note that this is the day when the report comes out, not the date of the fiscal end
    
#     mask = (
#         (df_daily['ticker'] == ticker) &
#         (df_daily['date'] >= start_date) &
#         (df_daily['date'] <= end_date)
#     )
#     sequence_data = df_daily[mask].sort_values('date')
    
#     # Handle missing days (days with no posts)
#     date_range = pd.date_range(start_date, end_date, freq='D')
#     sequence_data = sequence_data.set_index('date').reindex(date_range)
#     sequence_data['combined_text'] = sequence_data['combined_text'].fillna('')  # empty string for no posts
    
#     return sequence_data['combined_text'].tolist()

# def count_zeros_in_sequence(embedding_seq):
#     #this function checks the distribution of the number of posts in the 60-day look-back window
#     if isinstance(embedding_seq, torch.Tensor):
#         embedding_seq = embedding_seq.numpy()
    
#     zero_days = 0
#     for day_embedding in embedding_seq:
#         if np.abs(day_embedding).sum() < 0.01:  # Essentially zero
#             zero_days += 1
#     return zero_days

'''
PART III
This part does embedding
'''
def get_bert_embedding(tokenizer, model , text):
    if not text or text.strip() == '':  # handle empty days
        return torch.zeros(768)
    
    inputs = tokenizer(text, return_tensors='pt', 
                      truncation=True, max_length=512,
                      padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding
    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return embedding