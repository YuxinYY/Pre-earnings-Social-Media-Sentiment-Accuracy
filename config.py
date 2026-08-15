import argparse

argparser = argparse.ArgumentParser()
#data is not published, you may use your own datasets

# 大数据统一放在外部磁盘 T9；代码保持在 Mac 本机路径。
# 若 T9 未挂载，可临时改为 ./data_processing（但候选文件已迁移到 T9）。
DEFAULT_DATA_DIR = '/Volumes/T9/InterviewProjects/Hybrid_attention_network_learning/data_processing'

argparser.add_argument(
    '--data_dir',
    type=str,
    default=DEFAULT_DATA_DIR,
    help='外部磁盘上的数据目录（候选 CSV、股票面板、缓存、embedding 输出等）'
)

argparser.add_argument(
    '--submissionandcomments_dir',
    type=str,
    # 逗号分隔的多个候选 CSV（filter_candidate_posts.py 的产物）
    # 2021-2023 与 2024-2025 的候选已全部迁移到 T9
    default=(f'{DEFAULT_DATA_DIR}/submissions_2021_2023_candidates.csv,'
             f'{DEFAULT_DATA_DIR}/comments_2021_2023_candidates.csv,'
             f'{DEFAULT_DATA_DIR}/submissions_2024_2025_candidates.csv,'
             f'{DEFAULT_DATA_DIR}/comments_2024_2025_candidates.csv')

    # required=True,
    # help='Path to your private text data (not included due to license)'
)

argparser.add_argument(
    '--val_start_date',
    type=str,
    # 70/15/15 时间切分：样本区间由 pipeline.py 按 day_dict 文本覆盖决定
    # （约 2021-02 ~ 2025-12，约 1220 个交易日），三份样本量约 7:1.5:1.5
    default='2024-07-11',
    help='Train/Val 分界：anchor_date >= 该日期的样本进入验证集（约 70% 训练量）'
)

argparser.add_argument(
    '--test_start_date',
    type=str,
    default='2025-04-07',
    help='Val/Test 分界：anchor_date >= 该日期的样本进入测试集'
)

argparser.add_argument(
    '--sp_stocks_dir',
    type=str,
    default='matched_stocks.csv'
)

argparser.add_argument(
    '--stocks',
    type=str,
    default=f'{DEFAULT_DATA_DIR}/stocks_2020_2026.csv',
    help='股票面板：date, ticker, name, sector, RET, VOL（2020-01 ~ 2026-01，含 2024/2025）'
)

argparser.add_argument(
    '--daily_return_nyse',
    type=str,
    default='./nonuploads/daily_nyse_return.csv'
)

argparser.add_argument(
    '--eps_surprise_dir',
    type=str,
    default='training/datasets/earningreleasedatesandsurprises.csv'
    # required=True,
    # help='Path to your private earnings surprice data (not included due to license). May refer to :https://www.alphavantage.co/documentation/.'
)

argparser.add_argument(#for saving our training and testing datasets
    '--dataset_save_dir',
    type=str,
    default='training/datasets/'
)



args, unknown = argparser.parse_known_args()
# import sys
# if "__file__" in globals():  # Only parse when run as script, not when imported in notebook
#     args = argparser.parse_args()
# else:
#     args = argparser.parse_args([])  # Parse nothing (prevents Jupyter crash)

