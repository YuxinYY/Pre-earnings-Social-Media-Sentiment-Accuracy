"""
把 Reddit 文本和股票数据加工成 HAN 模型训练所需的 `.pt` 文件。

作用（行业/sector 级版本）:
    1. 读取抽样后的 Reddit CSV 和 `stocks.csv`。
    2. 用 FlashText + GLiNER 匹配帖子与 ticker，并把 ticker 映射到 GICS sector。
    3. 合并股票价格/成交量，构造文本统计特征。
    4. 用 FinBERT 生成 768 维 embedding。
    5. 按 `(sector, date)` 聚合帖子，构建 sector-day tensor。
    6. 用 sector 内股票等权收益率计算未来 20 日实现波动率（RV_20）作为标签。
    7. 生成时间序列样本（lookback W=20）并做时间切分。
    8. 输出 `day_dict.pt`, `train_samples.pt`, `val_samples.pt`, `test_samples.pt`, `config.pt`。

用法:
    python data_processing/scripts/pipeline.py

说明:
    - 输入路径由项目根目录下的 `config.py` 控制（`--submissionandcomments_dir`、`--stocks`）。
    - 需要从项目根目录运行，以保证相对路径正确。
    - 当前标签基于 sector 内股票等权组合；后续可切换为 SPDR sector ETF。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import torch

# 将项目根目录加入 sys.path，以便从根目录导入 config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
args = config.args
from helpers import (
    compute_future_realized_vol, perform_local_extraction, count_floats,
    count_keywords, batch_process_embeddings_stream, build_day_dict_compact,
    build_time_series_samples, temporal_train_val_test_split,
)


def build_sector_returns(df_stocks: pd.DataFrame, ret_col: str = "RET") -> pd.DataFrame:
    """
    对每个 sector 构建等权日收益率。
    返回 DataFrame: [sector, date, sector_RET]
    """
    df = df_stocks.copy()
    df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
    sector_ret = (
        df.groupby(['sector', 'date'])['RET']
        .mean()
        .reset_index()
        .rename(columns={'RET': 'sector_RET'})
    )
    return sector_ret


def main():
    # ---------- 1. 加载数据 ----------
    df_comments = pd.read_csv(args.submissionandcomments_dir)
    df_comments['id'] = df_comments['id'].astype(str)

    df_stocks = pd.read_csv(args.stocks)  # 列: date, ticker, name, sector, RET, VOL
    df_stocks['date'] = pd.to_datetime(df_stocks['date']).dt.date

    # 计算 sector 等权收益率和未来 RV
    sector_ret = build_sector_returns(df_stocks)
    sector_rv = compute_future_realized_vol(
        sector_ret,
        window=20,
        ret_col="sector_RET",
        ticker_col="sector",
        date_col="date"
    )

    my_tickers = list(df_stocks['ticker'].drop_duplicates())

    # filter 常见英文单词 ticker
    black_list = [
        'A', 'AI', 'AL', 'ALL', 'AN', 'AR', 'ARE', 'AS', 'AT', 'BE', 'BY', 'CEO', 'CO', 'COO',
        'DAY', 'FL', 'FOR', 'FOUR', 'HE', 'HI', 'I', 'IT', 'LOW', 'MA', 'MD', 'MN', 'MO', 'MS',
        'MT', 'NC', 'NE', 'NEW', 'NM', 'NOW', 'NYC', 'ONE', 'ONTO', 'OR', 'OUT', 'PAY', 'SC',
        'SD', 'SEE', 'SF', 'SUN', 'TWO', 'TX', 'UP', 'USA', 'WE', 'WY', 'YOU', 'EAT', 'SEND',
        'BEST', 'HOME', 'SAFE', 'TO', 'DO', 'IN', 'OF', 'GO', 'THE', 'US', 'SO'
    ]
    filtered_tickers = [t for t in my_tickers if t not in black_list]

    # 公司名(大写) -> ticker 映射，供 GLiNER 实体匹配
    names = dict(zip(df_stocks['name'].astype(str).str.upper(), df_stocks['ticker']))

    # ---------- 2. FlashText + GLiNER 匹配 ----------
    TEMP_RESULTS_FILE = "./data_processing/temp_matched.csv"

    # 如果 temp_matched.csv 比输入数据更新，直接复用，避免重复跑 GLiNER
    matched_df = None
    if os.path.exists(TEMP_RESULTS_FILE):
        input_mtime = os.path.getmtime(args.submissionandcomments_dir)
        stocks_mtime = os.path.getmtime(args.stocks)
        temp_mtime = os.path.getmtime(TEMP_RESULTS_FILE)
        if temp_mtime >= max(input_mtime, stocks_mtime):
            print(f"检测到较新的匹配缓存: {TEMP_RESULTS_FILE}，跳过 GLiNER 匹配")
            matched_df = pd.read_csv(TEMP_RESULTS_FILE)

    if matched_df is None:
        matched_df = perform_local_extraction(
            df_comments,
            TEMP_RESULTS_FILE,
            valid_tickers=filtered_tickers,
            names=names,
            black_list=set(black_list),
            gliner_model_path="./gliner_model",
        )

    # 只保留有效 ticker 的匹配结果
    matched_df = matched_df[matched_df['matched_ticker'].isin(filtered_tickers)]

    # 合并 Reddit 元数据
    df = pd.merge(matched_df, df_comments[['id', 'date', 'score']], on=['id'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    df.rename(columns={"matched_ticker": "ticker"}, inplace=True)

    # 合并股票数据，拿到 sector 和 VOL
    df = pd.merge(
        df,
        df_stocks[['ticker', 'date', 'sector', 'VOL']],
        on=['date', 'ticker'],
        how='inner'
    )

    # 结构化文本特征
    df['float_count'] = df['source_text'].apply(count_floats)
    df['keyword_count'] = df['source_text'].apply(count_keywords)
    df['word_count'] = df['source_text'].fillna('').apply(lambda x: len(str(x).split()))

    # 合并 sector RV 标签
    df = pd.merge(
        df,
        sector_rv.rename(columns={'RV_20': 'RV_20'}),
        left_on=['sector', 'date'],
        right_on=['sector', 'date'],
        how='inner'
    )
    df = df.dropna(subset=['RV_20'])

    print(f"匹配后样本量: {len(df)} 行, 涉及 {df['ticker'].nunique()} 个 ticker, {df['sector'].nunique()} 个 sector")

    # ---------- 3. FinBERT embedding ----------
    embedding_dir = "./data_processing/embedding_output"
    batch_process_embeddings_stream(
        df=df,
        output_dir=embedding_dir,
        chunk_size=10000,
        batch_size=256,
        model_path="./finbert_model",
    )

    df_result = pd.read_parquet(embedding_dir)
    df = df.reset_index(drop=True)
    df['original_index'] = df.index
    final_df = pd.merge(df, df_result, on='original_index', how='left')

    # ---------- 4. 构建样本并切分 ----------
    # 注意：key 从 (ticker, date) 改为 (sector, date)
    day_dict, E, D = build_day_dict_compact(final_df, ticker_col="sector")

    # sector 级标签数据，保证每天每个 sector 只有一个 label
    sector_label_df = sector_rv[['sector', 'date', 'RV_20']].dropna()
    samples = build_time_series_samples(
        sector_label_df,
        day_dict,
        W=20,
        label_col="RV_20",
        ticker_col="sector",
        date_col="date"
    )
    train_s, val_s, test_s = temporal_train_val_test_split(samples, "2023-08-01", "2023-11-01")

    # ---------- 5. 保存 ----------
    save_dir = "./data_processing"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving data to {save_dir} ...")
    torch.save(day_dict, os.path.join(save_dir, "day_dict.pt"))
    torch.save(train_s, os.path.join(save_dir, "train_samples.pt"))
    torch.save(val_s, os.path.join(save_dir, "val_samples.pt"))
    torch.save(test_s, os.path.join(save_dir, "test_samples.pt"))
    config_dict = {"E": E, "D": D, "L": 50}
    torch.save(config_dict, os.path.join(save_dir, "config.pt"))
    print("✅ Sector-level pipeline 完成")


if __name__ == '__main__':
    main()
