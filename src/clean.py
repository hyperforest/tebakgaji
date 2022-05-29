import pandas as pd

from modules.data import read_raw, clean_data

DATA_PATH = './datasets/data.html'
RAW_DATA_PATH = './datasets/raw_data.csv'
CLEAN_DATA_PATH ='./datasets/clean_data.csv'

def main():
    raw_data = read_raw(DATA_PATH)
    print('> Raw data shape:', raw_data.shape)

    clean_df = clean_data(
        raw_data,
        drop_low_outlier=False,
        drop_high_outlier=False
    )
    print('> Clean data shape:', clean_df.shape)

    raw_data.to_csv(RAW_DATA_PATH, index=False)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)

    print('> Raw and clean data saved to', RAW_DATA_PATH,
          'and', CLEAN_DATA_PATH)

if __name__ == '__main__':
    main()
