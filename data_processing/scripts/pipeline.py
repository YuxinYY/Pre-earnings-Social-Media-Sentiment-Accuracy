"""
把 Reddit 文本和股票数据加工成 HAN 模型训练所需的 `.pt` 文件。

作用（行业/sector 级版本）:
    1. 两遍扫描候选 Reddit CSV（submissions + comments，由 filter_candidate_posts.py 生成）:
       - 第一遍：分块做 FlashText + GLiNER 匹配，只保留匹配结果和文本统计特征（不存原文），
         然后按 (sector, date) 取 top DAY_CAP 条（day_dict 每天只用 top 50，先限额可避免
         对注定被丢弃的帖子跑 embedding，也控制内存）。
       - 第二遍：只对限额后幸存的帖子回扫原文，供 FinBERT embedding 使用。
    2. 合并股票价格，标签 = sector 等权组合未来 20 日收益 − 同期 S&P 500 收益（超额收益），>0 为跑赢。
    3. 按 `(sector, date)` 聚合帖子，构建 sector-day tensor。
    4. 生成时间序列样本（lookback W=20）并做时间切分。
    5. 输出 `day_dict.pt`, `train_samples.pt`, `val_samples.pt`, `test_samples.pt`, `config.pt`。

用法:
    python data_processing/scripts/pipeline.py

说明:
    - 输入路径由项目根目录下的 `config.py` 控制；`--submissionandcomments_dir` 支持
      逗号分隔的多个候选 CSV。
    - 需要从项目根目录运行，以保证相对路径正确。
    - 第一遍匹配结果缓存于 `data_processing/temp_matched_features.parquet`，
      比所有输入新时自动复用。
    - 重跑 embedding 前需手动清空 `data_processing/embedding_output/`（该目录按 chunk 断点续跑）。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import torch

# 将项目根目录加入 sys.path，以便从项目根目录导入 config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
args = config.args
from helpers import (
    compute_future_return, perform_local_extraction, count_floats,
    count_keywords, batch_process_embeddings_stream, build_day_dict_compact,
    build_time_series_samples, temporal_train_val_test_split, get_device,
)

# 分块读取候选 CSV 的块大小（控制内存）
CHUNK_SIZE = 1_000_000
# 每个 (sector, date) 最多保留多少条帖子进入 embedding；day_dict 最终只用 top L=50
DAY_CAP = 150

# filter 常见英文单词 ticker
BLACK_LIST = [
    'A', 'AI', 'AL', 'ALL', 'AN', 'AR', 'ARE', 'AS', 'AT', 'BE', 'BY', 'CEO', 'CO', 'COO',
    'DAY', 'FL', 'FOR', 'FOUR', 'HE', 'HI', 'I', 'IT', 'LOW', 'MA', 'MD', 'MN', 'MO', 'MS',
    'MT', 'NC', 'NE', 'NEW', 'NM', 'NOW', 'NYC', 'ONE', 'ONTO', 'OR', 'OUT', 'PAY', 'SC',
    'SD', 'SEE', 'SF', 'SUN', 'TWO', 'TX', 'UP', 'USA', 'WE', 'WY', 'YOU', 'EAT', 'SEND',
    'BEST', 'HOME', 'SAFE', 'TO', 'DO', 'IN', 'OF', 'GO', 'THE', 'US', 'SO'
]


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


def pass1_match_and_features(input_paths, df_stocks, filtered_tickers, names, cache_file):
    """
    第一遍扫描：分块匹配 + 计算文本统计特征，不保留原文。
    返回 DataFrame: [id, date, ticker, sector, score, float_count, keyword_count, word_count]
    """
    # 缓存复用条件：sidecar 记录的输入文件列表一致，且缓存比所有输入都新
    # （仅比较 mtime 不够：用较小输入跑出的缓存可能比大输入更新）
    meta_file = cache_file + ".meta.json"
    if os.path.exists(cache_file) and os.path.exists(meta_file):
        import json as _json
        with open(meta_file) as f:
            meta = _json.load(f)
        cache_mtime = os.path.getmtime(cache_file)
        inputs_mtime = max([os.path.getmtime(p) for p in input_paths] + [os.path.getmtime(args.stocks)])
        if meta.get("input_paths") == input_paths and cache_mtime >= inputs_mtime:
            print(f"检测到有效的匹配缓存: {cache_file}，跳过第一遍扫描")
            return pd.read_parquet(cache_file)
        print(f"缓存输入与当前不一致或已过期，重新跑第一遍扫描")

    # GLiNER 只加载一次，所有 chunk 复用（绝大多数行 direct hit 后根本不会用到它）
    from gliner import GLiNER
    gliner_path = "./gliner_model" if os.path.exists("./gliner_model") else "urchade/gliner_small-v2.1"
    device = get_device()
    print(f"Loading GLiNER from {gliner_path} on {device} ...")
    gliner_model = GLiNER.from_pretrained(gliner_path).to(device)
    gliner_model.eval()

    stocks_slim = df_stocks[['ticker', 'date', 'sector']]
    parts = []
    for path in input_paths:
        print(f"\n=== 第一遍扫描: {path} ===")
        for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_SIZE)):
            chunk['id'] = chunk['id'].astype(str)
            matched = perform_local_extraction(
                chunk,
                "./data_processing/temp_matched_chunk.csv",
                valid_tickers=filtered_tickers,
                names=names,
                black_list=set(BLACK_LIST),
                model=gliner_model,
            )
            if matched.empty:
                continue
            matched = matched[matched['matched_ticker'].isin(filtered_tickers)]
            if matched.empty:
                continue

            m = pd.merge(matched, chunk[['id', 'date', 'score']], on='id')
            m['date'] = pd.to_datetime(m['date']).dt.date
            m.rename(columns={"matched_ticker": "ticker"}, inplace=True)
            m = pd.merge(m, stocks_slim, on=['ticker', 'date'], how='inner')
            if m.empty:
                continue

            m['float_count'] = m['source_text'].apply(count_floats)
            m['keyword_count'] = m['source_text'].apply(count_keywords)
            m['word_count'] = m['source_text'].fillna('').apply(lambda x: len(str(x).split()))
            # 不保留 source_text：16GB 内存放不下全量匹配结果的原文
            parts.append(m.drop(columns=['source_text']))
            print(f"  [chunk {i}] 累计匹配行数: {sum(len(p) for p in parts)}")

    feat = pd.concat(parts, ignore_index=True)
    feat.to_parquet(cache_file, index=False)
    import json as _json
    with open(cache_file + ".meta.json", "w") as f:
        _json.dump({"input_paths": input_paths}, f)
    print(f"第一遍扫描完成: {len(feat)} 匹配行, 已缓存到 {cache_file}")
    return feat


def pass2_fetch_survivor_texts(input_paths, survivor_ids):
    """
    第二遍扫描：只对限额后幸存的 id 回扫原文，重建 source_text。
    返回 DataFrame: [id, source_text]
    """
    texts = []
    for path in input_paths:
        print(f"\n=== 第二遍扫描: {path} ===")
        for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_SIZE)):
            chunk['id'] = chunk['id'].astype(str)
            hit = chunk[chunk['id'].isin(survivor_ids)]
            if hit.empty:
                continue
            texts.append(hit[['id', 'title', 'selftext', 'body']])
            print(f"  [chunk {i}] 累计回收文本: {sum(len(t) for t in texts)}")

    df_text = pd.concat(texts, ignore_index=True).drop_duplicates('id')
    df_text['source_text'] = (
        df_text['title'].fillna('') + ' ' +
        df_text['selftext'].fillna('') + ' ' +
        df_text['body'].fillna('')
    ).str.strip()
    return df_text[['id', 'source_text']]


def main():
    # ---------- 0. 输入 ----------
    input_paths = [p.strip() for p in args.submissionandcomments_dir.split(',') if p.strip()]
    print(f"输入候选 CSV: {input_paths}")

    df_stocks = pd.read_csv(args.stocks)  # 列: date, ticker, name, sector, RET, VOL
    df_stocks['date'] = pd.to_datetime(df_stocks['date']).dt.date

    # ---------- 1. 标签：sector 超额收益 ----------
    sector_ret = build_sector_returns(df_stocks)
    sector_fut = compute_future_return(
        sector_ret,
        window=20,
        ret_col="sector_RET",
        ticker_col="sector",
        date_col="date"
    )

    # S&P 500 未来 20 日收益，超额收益 = sector 收益 − 指数收益
    df_sp500 = pd.read_csv("./data_processing/sp500.csv")  # 列: date, SP500_RET
    df_sp500['date'] = pd.to_datetime(df_sp500['date']).dt.date
    df_sp500['index'] = 'SP500'  # compute_future_return 需要 ticker 列
    sp500_fut = compute_future_return(
        df_sp500,
        window=20,
        ret_col="SP500_RET",
        ticker_col="index",
        date_col="date"
    ).rename(columns={'future_20d_ret': 'sp500_future_20d_ret'})

    sector_label = pd.merge(sector_fut, sp500_fut[['date', 'sp500_future_20d_ret']], on='date', how='left')
    sector_label['exret_20'] = sector_label['future_20d_ret'] - sector_label['sp500_future_20d_ret']
    sector_label = sector_label.dropna(subset=['exret_20'])

    my_tickers = list(df_stocks['ticker'].drop_duplicates())
    filtered_tickers = [t for t in my_tickers if t not in BLACK_LIST]

    # 公司名(大写) -> ticker 映射，供 GLiNER 实体匹配
    names = dict(zip(df_stocks['name'].astype(str).str.upper(), df_stocks['ticker']))

    # ---------- 2. 第一遍：分块匹配 + 特征（无原文） ----------
    feat = pass1_match_and_features(
        input_paths, df_stocks, filtered_tickers, names,
        cache_file="./data_processing/temp_matched_features.parquet",
    )
    print(f"匹配行数: {len(feat)}, 涉及 {feat['ticker'].nunique()} 个 ticker, {feat['sector'].nunique()} 个 sector")

    # ---------- 3. 按 (sector, date) 限额 ----------
    # 排序规则与 build_day_dict_compact 一致，保证最终 top-L 选择不受限额影响
    feat = feat.sort_values(
        ['sector', 'date', 'keyword_count', 'word_count', 'score'],
        ascending=[True, True, False, False, False]
    )
    capped = feat.groupby(['sector', 'date'], sort=False).head(DAY_CAP).reset_index(drop=True)
    del feat
    print(f"限额后保留: {len(capped)} 行 (每天每 sector 最多 {DAY_CAP} 条)")

    # ---------- 4. 第二遍：只对幸存帖子回扫原文 ----------
    survivor_ids = set(capped['id'])
    df_text = pass2_fetch_survivor_texts(input_paths, survivor_ids)
    df = pd.merge(capped, df_text, on='id', how='left')
    print(f"回扫文本后: {len(df)} 行, 缺文本: {df['source_text'].isna().sum()} 行")

    # 合并 sector 超额收益标签（仅用于把文本行过滤到有标签的日期）
    df = pd.merge(
        df,
        sector_label[['sector', 'date', 'exret_20']],
        on=['sector', 'date'],
        how='inner'
    )
    df = df.dropna(subset=['exret_20'])

    # ---------- 5. FinBERT embedding ----------
    # 同一帖子可能匹配多个 ticker（每个 ticker 一行），按 id 去重后再做 embedding，
    # 最后把 embedding 合并回所有匹配行，避免对同一文本重复推理
    df_unique = df.drop_duplicates('id').reset_index(drop=True)
    print(f"去重后待 embedding 帖子数: {len(df_unique)} (匹配行数: {len(df)})")

    embedding_dir = "./data_processing/embedding_output"
    batch_process_embeddings_stream(
        df=df_unique,
        output_dir=embedding_dir,
        chunk_size=10000,
        batch_size=256,
        model_path="./finbert_model",
    )

    df_result = pd.read_parquet(embedding_dir)
    df_unique['original_index'] = df_unique.index
    df_unique = pd.merge(df_unique, df_result, on='original_index', how='left')
    final_df = pd.merge(df, df_unique[['id', 'embedding']], on='id', how='left')

    # ---------- 6. 构建样本并切分 ----------
    # 注意：key 从 (ticker, date) 改为 (sector, date)
    day_dict, E, D = build_day_dict_compact(final_df, ticker_col="sector")

    # sector 级标签数据，保证每天每个 sector 只有一个 label
    sector_label_df = sector_label[['sector', 'date', 'exret_20']].dropna()
    samples = build_time_series_samples(
        sector_label_df,
        day_dict,
        W=20,
        label_col="exret_20",
        ticker_col="sector",
        date_col="date",
        threshold=0.0,  # 超额收益有自然零点：>0 即跑赢大盘
    )
    train_s, val_s, test_s = temporal_train_val_test_split(samples, args.val_start_date, args.test_start_date)

    # ---------- 7. 保存 ----------
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
