import argparse
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import os
import pickle

TICKERS = [
'MSFT',
'AAPL',
'TSLA',
'FB',
'F',
'KO',
'CMCSA',
'NFLX',
'INTU',
'NVDA',
'BABA',
'EA',
'ADBE',
'ADSK',
'GOOGL',
'INTC',
'COLM',
'CSCO',
'PEP',
'AMZN',
'SPOT',
'NIO',
'VTI',
'VOO',
'AMAT',
'BYDDY',
'GM',
'FCAU',
'GOOG',
'UBER',
'LYFT',
'T',
'FDX',
'BIDU',
'SNAP',
'QCOM',
'BYND',
'DIS',
'FIT',
'GPRO',
]

def parse_args():
  """ Parse arguments for program
  """
  parser = argparse.ArgumentParser(description="Program for analyzing stock")
  subparsers = parser.add_subparsers(dest='sub_cmd',required=True)

  plot_parser = subparsers.add_parser('plot', help='plot the stock data')
  plot_parser.add_argument('-s','--start',default='2018/1/1',help="start date in the format YYYY/mm/dd")
  plot_parser.add_argument('-e','--end',default='2019/12/26',help="end date in the format YYYY/mm/dd")
  plot_parser.set_defaults(func='plot')

  return parser.parse_args()

def download_data(start = dt.datetime(2000,1,1), end = dt.datetime(2019,12,26), tickers = TICKERS, force = False):
    """ Download data from yahoo for provided tickers
    """
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)) or force:
            try:
                print('Downloading {}'.format(ticker))
                df = web.DataReader(ticker,'yahoo',start,end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            except Exception as e:
                print('Failed to download {}:\n\t{}'.format(ticker,e))
        else:
            print('Already have {}'.format(ticker))

def load_data(tickers = TICKERS):
    main_df = pd.DataFrame()
    for ticker in tickers:
        try:
            print('Compiling {}'.format(ticker))
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col = 'Date',parse_dates = ['Date'])
            df.rename(columns = {'Adj Close':ticker}, inplace = True)
            df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df,how='outer')
        except Exception as e:
            print('Failed to compile {}:\n\t{}'.format(ticker,e))

    return main_df

def normalize(df_in):

    # Get first non null indexes
    idx = df_in.notnull().idxmax()

    # Copy
    df = df_in.copy()

    # Normalize each column
    for col in df:
        df[col] = df[col]/df[col].loc[idx[col]]

    return df


def get_timeframe(df,start,end):

    # Create mask between time
    mask = (df.index > start) & (df.index < end)

    return df.loc[mask]

def create_value_legend(tickers,values):

    leg = []

    for i in range(len(tickers)):
        leg.append('{} = {:.2f}'.format(tickers[i],values[i]))

    return leg

def plot_data(start,end):
    # Set start and end
    start = dt.datetime.strptime(start,'%Y/%m/%d')
    end = dt.datetime.strptime(end,'%Y/%m/%d')

    # Download data
    download_data()

    # Load the data
    main_df = load_data()

    # Get subset
    df = get_timeframe(main_df,start,end)

    # normalize
    df = normalize(df)

    # Sort
    df = df.sort_values(df.last_valid_index(), ascending = False, axis=1)

    # Get last values
    last_values = df.iloc[-1]

    # Get legend
    leg = create_value_legend(list(df),last_values)

    # Plot
    plt.plot(df)
    plt.xlabel('date')
    plt.ylabel('gain')
    plt.grid('True')
    plt.legend(leg)

    plt.show()

if __name__ == "__main__":
    args = parse_args()

    if args.func == 'plot':
        plot_data(args.start, args.end)
