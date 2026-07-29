"""
从历史 cashtag 高频 ticker 中抓取股票日线数据，产出 pipeline 所需的 stocks.csv。

作用:
    - 读取 `data_processing/ticker_sector_map.csv`（由 cashtag 扫描生成）。
    - 取按 mention 数排序的 top N 只非 ETF 股票，作为新的 ticker universe。
    - 用 yfinance 抓取每只股票的日线价格和成交量，并尝试获取公司名与 sector。
    - 输出 `stocks.csv`：`date, ticker, name, sector, RET, VOL`。

输出列:
    - date: 交易日期 (YYYY-MM-DD)
    - ticker: 股票代码
    - name: 公司简称（yfinance shortName，若缺失则回退为 ticker）
    - sector: GICS sector（来自 ticker_sector_map.csv）
    - RET: 日收盘收益率（adjusted close pct_change）
    - VOL: 成交量

用法:
    python data_processing/scripts/fetch_stock_data.py \
        --start 2021-01-01 --end 2023-12-31 \
        --output stocks.csv \
        --top_n 300

说明:
    - 如果 `ticker_sector_map.csv` 不存在，脚本会回退到最初硬编码的 20 个 WSB 热门 ticker，
      以保证 pipeline 在小规模验证时仍可运行。
"""

import argparse
import os
import time
import pandas as pd
import yfinance as yf

# 初始 20 个 WSB 热门 ticker，作为 fallback
DEFAULT_TICKERS = {
    'GME': 'GameStop',
    'AMC': 'AMC Entertainment',
    'TSLA': 'Tesla',
    'AAPL': 'Apple',
    'NVDA': 'NVIDIA',
    'PLTR': 'Palantir',
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    'AMD': 'Advanced Micro Devices',
    'SPY': 'S&P 500 ETF',
    'BB': 'BlackBerry',
    'NOK': 'Nokia',
    'SOFI': 'SoFi Technologies',
    'HOOD': 'Robinhood',
    'COIN': 'Coinbase',
    'MARA': 'Marathon Digital',
    'RIVN': 'Rivian',
    'LCID': 'Lucid Motors',
    'F': 'Ford',
    'INTC': 'Intel',
}

# 常见 ETF/指数产品，不纳入个股 universe
KNOWN_ETFS = {
    'SPY', 'QQQ', 'SLV', 'GLD', 'VOO', 'VTI', 'ARKK', 'IWM',
    'TQQQ', 'SQQQ', 'UVXY', 'VXX', 'SPXL', 'SPXS', 'QQQM', 'DIA',
    'XLF', 'XLK', 'XLE', 'XLU', 'XLI', 'XLP', 'XLY', 'XLB', 'XLRE', 'XLC', 'XBI'
}


def load_universe(universe_file: str, top_n: int):
    """从 ticker_sector_map.csv 加载 top N ticker，返回 [(ticker, sector), ...]。"""
    if not os.path.exists(universe_file):
        print(f"⚠️ Universe 文件不存在: {universe_file}，回退到默认 20 个 ticker")
        return [(t, 'MISSING') for t in DEFAULT_TICKERS.keys()]

    df = pd.read_csv(universe_file)
    df = df[df['ticker'].notna()]
    # 去掉 ETF 和 sector 缺失的行
    df = df[~df['ticker'].isin(KNOWN_ETFS)]
    df = df[df['sector'].notna() & (df['sector'] != 'MISSING')]
    df = df.nlargest(top_n, 'total_mentions')
    print(f"从 {universe_file} 读取 top {top_n} ticker: {len(df)} 个")
    return list(zip(df['ticker'], df['sector']))


def fetch_name_sector(ticker: str, fallback_sector: str):
    """尝试获取 yfinance shortName 和 sector；失败则回退。"""
    name = ticker
    sector = fallback_sector
    try:
        info = yf.Ticker(ticker).info
        short_name = info.get('shortName')
        if short_name:
            name = short_name
        yf_sector = info.get('sector')
        if yf_sector:
            sector = yf_sector
    except Exception:
        pass
    return name, sector


def main():
    parser = argparse.ArgumentParser(description="抓取扩展 ticker universe 的股票日线数据")
    parser.add_argument('--start', default='2021-01-01', help='开始日期')
    parser.add_argument('--end', default='2023-12-31', help='结束日期')
    parser.add_argument('--output', default='stocks.csv', help='输出 CSV')
    parser.add_argument('--universe_file', default='data_processing/ticker_sector_map.csv',
                        help='ticker_sector_map.csv 路径')
    parser.add_argument('--top_n', type=int, default=300, help='取 mention 数前 N 的 ticker')
    parser.add_argument('--sleep', type=float, default=0.12, help='yfinance 请求间隔（秒）')
    args = parser.parse_args()

    universe = load_universe(args.universe_file, args.top_n)

    rows = []
    for ticker, sector in universe:
        print(f"Fetching {ticker} ...")
        name, sector_final = fetch_name_sector(ticker, sector)

        try:
            hist = yf.Ticker(ticker).history(start=args.start, end=args.end, auto_adjust=True)
        except Exception as e:
            print(f"  ⚠️ {ticker} 下载失败: {e}")
            continue

        if hist.empty:
            print(f"  ⚠️ {ticker} 无数据，跳过")
            continue

        hist = hist.reset_index()
        hist['RET'] = hist['Close'].pct_change()
        hist = hist.dropna(subset=['RET'])

        for _, r in hist.iterrows():
            rows.append({
                'date': r['Date'].strftime('%Y-%m-%d'),
                'ticker': ticker,
                'name': name,
                'sector': sector_final,
                'RET': r['RET'],
                'VOL': r['Volume'],
            })

        time.sleep(args.sleep)

    if not rows:
        raise RuntimeError("没有抓到任何股票数据，请检查 universe_file 或网络连接。")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"✅ 已写出 {args.output}: {len(df)} 行, {df['ticker'].nunique()} 个 ticker, "
          f"{df['sector'].nunique()} 个 sector, 日期范围 {df['date'].min()} ~ {df['date'].max()}")


if __name__ == '__main__':
    main()
