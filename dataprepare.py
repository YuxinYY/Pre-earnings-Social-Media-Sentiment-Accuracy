import pandas as pd
import numpy as np
import re
from gliner import GLiNER
from tqdm import tqdm
import time
import json
import re
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

import sys
from pathlib import Path

import config
args = config.args

from helper import generate_full_text, build_cleaned_flashtext_processor, local_process_comment_no_llm_or_embedding, create_sequences, get_bert_embedding

#Step 1: loading data
#complete comment data 
df_comments = pd.read_csv(args.submissionandcomments_dir)
#stocks we care about
df_stocks = pd.read_csv(args.stocks)

my_tickers = list(df_stocks.iloc[:,3].drop_duplicates())

#load security names from sec
import requests
import json
import re

url = "https://www.sec.gov/files/company_tickers.json"
resp = requests.get(url, headers={"User-Agent": "YOUR USER AGENT"})
data = resp.json()

df_map = pd.DataFrame.from_dict(data, orient="index")
df_map["cik_str"] = df_map["cik_str"].astype(str).str.zfill(10)
matched = df_map[df_map['ticker'].isin(my_tickers)]
words_to_remove = ['inc', '.', 'corp']
pattern = r'\b(' + '|'.join(words_to_remove) + r')\b'

#normalize text:
matched['full_text'] = matched.apply(generate_full_text, axis = 1)
# matched['name'] = matched['title'].str.replace(pattern, '', flags=re.IGNORECASE, regex=True)


#Step 2: stock matching using NER, finding out which stock a particular line of text might have mentioned
# STOCK_DB_PATH = r"nonuploads\matched_stocks_1119.csv" #matched
COMMENTS_CHUNK_FILE = df_comments
TEMP_RESULTS_FILE = r"Thepathyousaveintermediateresults.csv"
FINAL_RESULTS_FILE = r"Thepathyousave.csv" # Output file for final results
GLINER_MODEL = "urchade/gliner_small-v2.1" #CPU and light weight mode, refer to: https://github.com/urchade/GLiNER


TICKER_BLACKLIST = {# this is a list of tickers I forcifully remove to reduce mismatch; you may build your own
    'A',     # Agilent Technologies (proposition)
    'IT',    # Gartner (proposition;)
    'BE',    # Bloom Energy 
    'OR',    # Oracle
    'TO',    
    'IN',    
    'OF',    
    'DO',    
    'GO',    
    'ARE',   
    'FOR',   
    'AND',   
    'THE',   
    'YOU',   #
    'I',     
    'MY',    
    'BY',    
    'US',     
    'NOW',   
    'AM',    
    'AN',    
    'SO',    
    'UP'     
}

gliner_model = GLiNER.from_pretrained(GLINER_MODEL, resume_download=True)
df_stocks_for_linking = matched
keyword_processor, VALID_TICKERS = build_cleaned_flashtext_processor(matched, TICKER_BLACKLIST)
print(f"✅ FlashText size: {len(keyword_processor)}")

# Create a map for GLiNER entities to link back to tickers via exact name match
company_name_to_ticker_map = {}
for _, row in df_stocks_for_linking.iterrows():
    ticker = str(row['ticker']).upper()
    full_text_parts = [part.strip().upper() for part in str(row['full_text']).split(',') if part.strip()]
    
    # Add original title and cleaned alias from full_text to the lookup map
    # full_text is like: "ELI LILLY & Co, ticker: LLY, ELI LILLY"
    if len(full_text_parts) > 0: # Original title
        company_name_to_ticker_map[full_text_parts[0]] = ticker
    if len(full_text_parts) > 2: # Cleaned alias
        company_name_to_ticker_map[full_text_parts[2]] = ticker
    company_name_to_ticker_map[ticker] = ticker # Ensure ticker itself is in the map

# For quick lookup, create a set of all normalized company names/aliases
normalized_company_names_set = set(company_name_to_ticker_map.keys())

#running the matching
if 'combined_text' not in df_comments.columns:
        df_comments['combined_text'] = (
            df_comments['title'].fillna('') + ' ' + #these columns are innate to the specific datasets
            df_comments['selftext'].fillna('') + ' ' +
            df_comments['body'].fillna('')
        ).str.strip()
all_final_results_flat = []
total_rows = len(df_comments)
save_interval = max(1, int(total_rows * 0.05))
last_save_row_index = 0

df_comments['id'] = df_comments['id'].astype(str)

with tqdm(total=total_rows, desc="Local Entity Processing (FlashText + GLiNER)") as pbar:
        for index, row in df_comments.iterrows():
            matched_tickers = local_process_comment_no_llm_or_embedding(
                row['combined_text'], 
                row['id']
            )

            if matched_tickers:
                for ticker in matched_tickers:
                    all_final_results_flat.append({
                        "id": row['id'],
                        "source_text": row['combined_text'],
                        "matched_ticker": ticker
                    })
            
            pbar.update(1)
            
            current_row_index = pbar.n
            if (current_row_index >= last_save_row_index + save_interval) or current_row_index == total_rows:
                df_temp = pd.DataFrame(all_final_results_flat)
                #saving intermediate results
                df_temp.to_csv(TEMP_RESULTS_FILE, index=False)
                print('Progress', current_row_index, 'rows，saved to TEMP_RESULTS_FILE')
                last_save_row_index = current_row_index
                pbar.write(f"\n⏳ Has processed {current_row_index} / {total_rows} of ({current_row_index/total_rows:.2%}), intermediate results saved to {TEMP_RESULTS_FILE}")
ner_data = pd.DataFrame(all_final_results_flat)
ner_data.to_csv(FINAL_RESULTS_FILE, index=False)
    
# clearing intermediate results
if os.path.exists(TEMP_RESULTS_FILE):
    os.remove(TEMP_RESULTS_FILE)

#Step3: Merging with daily returns data
matched_posts = pd.merge(ner_data, df_comments[['id', 'date']], on = ['id'], how = 'left')
mentions_df = pd.merge(matched[['ticker']].drop_duplicates(), matched_posts, left_on = ['ticker'], right_on = ['matched_ticker'], how='left')
mentions_df['date'] = pd.to_datetime(mentions_df['date']).dt.date

#loading daily
daily_ret = pd.read_csv(args.daily_return_nyse)
daily_ret['date'] = pd.to_datetime(daily_ret['date'], format=None, dayfirst=True)
daily_ret['RET'] = pd.to_numeric(daily_ret['RET'], errors='coerce')
daily_ret.rename(columns={"TICKER":"ticker"}, inplace=True)

mentions_df['date'] = pd.to_datetime(mentions_df['date'])
mentions_df = mentions_df.sort_values('date')

daily_ret['reddit_sequence'] = daily_ret.apply(
    lambda row: create_sequences(row, mentions_df, 8), axis=1
)

#Step 4: conduct embedding on text sequences using finBert
tokenizer = AutoTokenizer.from_porganizationsretrained('ProsusAI/finbert')
model = AutoModel.from_pretrained('ProsusAI/finbert')
# Apply to each day in each sequence
def embed_sequence(text_sequence):
    embeddings = [get_bert_embedding(tokenizer, model, text) for text in text_sequence]
    return torch.stack(embeddings) 

daily_ret['embedding_sequence'] = daily_ret['reddit_sequence'].apply(embed_sequence)
# ticker | date          | daily_return | embedding_sequence
# TSLA   | 2020-02-15    | +5%          | Tensor(60, 768)
# AAPL   | 2020-02-20    | -2%          | Tensor(60, 768)


#Last Step: preparing train, validation and test data
daily_ret = daily_ret.sort_values('date').reset_index(drop=True)

# Define your date splits 
train_end_date = pd.to_datetime('2024-06-30') #the current date value is only a showcase
val_end_date = pd.to_datetime('2024-09-30')

# Create boolean masks based on dates
train_mask = daily_ret['date'] <= train_end_date
val_mask = (daily_ret['date'] > train_end_date) & (daily_ret['date'] <= val_end_date)
test_mask = daily_ret['date'] > val_end_date

# Split data using date-based masks
train_returns = daily_ret[train_mask]
val_returns = daily_ret[val_mask]
test_returns = daily_ret[test_mask]

# Stack embeddings into tensors
X_train = torch.stack(train_returns['embedding_sequence'].tolist())
y_train = torch.tensor((train_returns['RET'] > 0).values, dtype=torch.float32) #we care about raw return for now

X_val = torch.stack(val_returns['embedding_sequence'].tolist())
y_val = torch.tensor((val_returns['RET'] > 0).values, dtype=torch.float32)

X_test = torch.stack(test_returns['embedding_sequence'].tolist())
y_test = torch.tensor((test_returns['RET'] > 0).values, dtype=torch.float32)

save_path = args.dataset_save_dir
dir = Path(save_path)
dir.mkdir(parents=True, exist_ok=True)

np.save(save_path / 'X_train.npy', X_train.cpu().numpy())
print(f"✓ Saved X_train.npy (shape: {X_train.shape})")
np.save(save_path / 'y_train.npy', y_train.cpu().numpy())
print(f"✓ Saved y_train.npy (shape: {y_train.shape})")
np.save(save_path / 'X_val.npy', X_val.cpu().numpy())
print(f"✓ Saved X_val.npy (shape: {X_val.shape})")
np.save(save_path / 'y_val.npy', y_val.cpu().numpy())
print(f"✓ Saved y_val.npy (shape: {y_val.shape})")
np.save(save_path / 'X_test.npy', X_test.cpu().numpy())
print(f"✓ Saved X_test.npy (shape: {X_test.shape})")
np.save(save_path / 'y_test.npy', y_test.cpu().numpy())
print(f"✓ Saved y_test.npy (shape: {y_test.shape})")
print("\nAll files saved successfully!")
