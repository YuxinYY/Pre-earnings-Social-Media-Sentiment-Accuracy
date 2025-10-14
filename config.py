import argparse

argparser = argparse.ArgumentParser()
#data is not published, you may use your own datasets

argparser.add_argument(
    '--submissionandcomments_dir',
    type=str,
    required=True,
    help='Path to your private text data (not included due to license)'
)

argparser.add_argument(
    '--sp_stocks_dir',
    type=str,
    default='matched_stocks.csv'
)

argparser.add_argument(
    '--eps_surprise_dir',
    type=str,
    required=True,
    help='Path to your private earnings surprice data (not included due to license). May refer to :https://www.alphavantage.co/documentation/.'
)

argparser.add_argument(
    '--dataset_save_dir',
    type=str,
    default='training/datasets/'
)

args = argparser.parse_args()