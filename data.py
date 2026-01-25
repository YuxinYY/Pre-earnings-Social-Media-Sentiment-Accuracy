import pandas as pd
import dask.dataframe as dd
import numpy as np
import re
import sys
from pathlib import Path
import config
args = config.args
from helper import compute_future_realized_vol, perform_local_extraction, count_floats, count_keywords, batch_process_embeddings_stream, build_day_dict_compact, build_time_series_samples, temporal_train_val_test_split

#loading social media text
df_comments = pd.read_csv(args.submissionandcomments_dir)
df_stocks = pd.read_csv(args.stocks) #the file should have date, stock ticker, stock name, and daily return

rv5 = compute_future_realized_vol(df_stocks, window = 5) #future 5-day realized volatility

my_tickers = list(df_stocks.iloc[:,3].drop_duplicates()) # the list of stocks we should care about

#filter 
filtered_tickers = []
black_list = [
    'A', 'AI', 'AL', 'ALL', 'AN', 'AR', 'ARE', 'AS', 'AT', 'BE', 'BY', 'CEO', 'CO', 'COO', 'DAY', 'FL', 'FOR', 'FOUR', 'HE', 'HI', 'I', 'IT', 'LOW', 'MA', 'MD', 'MN', 'MO', 'MS', 'MT', 'NC', 'NE', 'NEW', 'NM', 'NOW', 'NYC', 'ONE', 'ONTO', 'OR', 'OUT', 'PAY', 'SC', 'SD', 'SEE', 'SF', 'SUN', 'TWO', 'TX', 'UP', 'USA', 'WE', 'WY', 'YOU', 'EAT', 'PAY', 'SEE', 'SEND', 'BEST', 'HOME', 'LOW', 'SAFE', 'OR', 'TO', 'DO', 'IN', 'OF', 'GO', 'THE', 'BY', 'US', 'NOW', 'SO', 'UP'

]
for t in my_tickers:
    if t not in black_list:
        filtered_tickers.append(t)

#mathcing individual social media posts with stocks so that we preserve relevant soical media posts
from gliner import GLiNER
from tqdm import tqdm
import time
import json
import torch
import os
import pyarrow.parquet as pq

COMMENTS_CHUNK_FILE = df_comments
FINAL_RESULTS_FILE = "" #path for final matched results
TEMP_RESULTS_FILE = ""

#the gliner model is 
gliner_model = "urchade/gliner_small-v2.1"
if torch.cuda.is_available():
    print(f"GLiNER loaded on GPU: {gliner_model.device}")
else:
    print("No CUDA/GPU detected, GliNer running on CPU.")

matched_df = perform_local_extraction(COMMENTS_CHUNK_FILE)

#merging matched results with stock volatility
matched_df = matched_df[matched_df['matched_ticker'].isin(filtered_tickers)]

df = pd.merge(matched_df, df_comments[['id', 'date', 'score']], on = ['id']) #score is the net upvotes from Reddit
df['date'] = pd.to_datetime(df['date']).dt.date
df.rename(columns = {"matched_ticker":"ticker"}, inplace = True)
df = pd.merge(df, df_stocks[['ticker', 'date', 'VOL']], on = ['date', 'ticker'], how = 'inner')

#structured features
#     #1.
# scr14 = lastNday_avg_score(df, 14)
#     #2. 
df['float_count'] = df['source_text'].apply(count_floats) #df has a "source_text" column after matching
df['keyword_count'] = df['source_text'].apply(count_keywords)
df['word_count'] = df['source_text'].fillna('').apply(lambda x: len(str(x).split()))

#computing labels: realized volatility
df = pd.merge(df, rv5, on = ['ticker', 'date'], how = 'inner')

#conduct embedding on post level: 'source_text'
embedding_dir = ""
df_with_embeddings = batch_process_embeddings_stream( #this is saved using parquet
    df=df,
    output_dir=embedding_dir, 
    chunk_size=10000, 
    batch_size=256,    
    model_path="./finbert_model" #local model; choose the model you prefer
)

df_result = dd.read_parquet(embedding_dir) #2 columns: original_index and embedding
df = df.reset_index(drop=True)
df_result = df_result.reset_index(drop=True)
df_dask = dd.from_pandas(df, npartitions=df_result.npartitions)
final_df = dd.concat([df_result, df_dask], axis=1) 


#splitting data to prepare for training
day_dict, E, D = build_day_dict_compact(final_df)
samples = build_time_series_samples(final_df, day_dict, W=20, label_col="RV_5")
train_s, val_s, test_s = temporal_train_val_test_split(samples, "2024-06-30", "2024-09-30")

#saving data
save_dir = "./data processing" 
os.makedirs(save_dir, exist_ok=True)
print(f"Saving data to {save_dir} ...")
torch.save(day_dict, os.path.join(save_dir, "day_dict.pt"))
torch.save(train_s, os.path.join(save_dir, "train_samples.pt"))
torch.save(val_s, os.path.join(save_dir, "val_samples.pt"))
torch.save(test_s, os.path.join(save_dir, "test_samples.pt"))
config = {"E": E, "D": D, "L": 50} 
torch.save(config, os.path.join(save_dir, "config.pt"))