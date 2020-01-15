import argparse
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import os
import pickle

def parse_args():
  """ Parse arguments for program
  """
  parser = argparse.ArgumentParser(description="Program for analyzing stock")
  subparsers = parser.add_subparsers(dest='sub_cmd',required=True)

  plot_parser = subparsers.add_parser('plot', help='plot the stock data')
  plot_parser.add_argument('-s','--start',default='2018/1/1',help="start date in the format YYYY/mm/dd")
  plot_parser.add_argument('-e','--end',default=dt.datetime.today().strftime("%Y/%m/%d"),help="end date in the format YYYY/mm/dd")
  plot_parser.add_argument('-t','--tickers', nargs='*', help='List of tickers')
  plot_parser.set_defaults(func='plot')

  update_parser = subparsers.add_parser('update',help='update the stock data')
  update_parser.set_defaults(func='update')

  download_parser = subparsers.add_parser('download',help='download the stock data')
  download_parser.add_argument('-s','--start',default='2000/1/1',help="start date in the format YYYY/mm/dd")
  download_parser.add_argument('-e','--end',default=dt.datetime.today().strftime("%Y/%m/%d"),help="end date in the format YYYY/mm/dd")
  download_parser.add_argument('-f','--force',action='store_true',help="force download")
  download_parser.set_defaults(func='download')

  return parser.parse_args()

def get_ticker_data():
    """ Get ticker data from csv
    """
    df = pd.read_csv('ticker_data.csv', index_col = 'Ticker')
    return df

def get_tickers():
    """ Get tickers from ticker data
    """
    df = get_ticker_data()
    return df.index.tolist()


def update_data(tickers = get_tickers()):
    """ Update the data to the current date
    """
    for ticker in tickers:
        try:
            print('Updating {}'.format(ticker))

            # Load old data
            df_old = pd.read_csv('stock_dfs/{}.csv'.format(ticker),index_col = 'Date',parse_dates = ['Date'])

            # Get last date
            last_date = df_old.index[-1]

            # Download new data
            start = last_date
            end = dt.datetime.today()
            df_new = web.DataReader(ticker,'yahoo',start,end)

            # Append
            df = df_old.append(df_new)

            # Remove duplicates
            # Dropping duplicates using index doesn't work
            df = df.reset_index().drop_duplicates(subset='Date').set_index('Date')

            # Save
            df.to_csv('stock_dfs/{}.csv'.format(ticker))

        except Exception as e:
            print('Failed to update {}:\n\t{}'.format(ticker,e))
    
def download_data(start, end, tickers = get_tickers(), force = False):
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

def load_data(tickers = get_tickers()):
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

    # Store normalization factors
    factors = []

    # Get first non null indexes
    idx = df_in.notnull().idxmax()

    # Copy
    df = df_in.copy()

    # Normalize each column
    for col in df:
        factor = df[col].loc[idx[col]]
        factors.append(factor)
        df[col] = df[col]/factor

    return df, factors

def get_timeframe(df,start,end):
    """ Get timeframe section from df
    """

    # Create mask between time
    mask = (df.index > start) & (df.index < end)

    return df.loc[mask]

def create_legend(tickers,last_values,factors):

    leg = []
    ticker_data = get_ticker_data()

    for i in range(len(tickers)):
        title = ticker_data.loc[tickers[i],'Title']
        leg.append('{} ({}): ({:.0f}/{:.0f}) {:.2f}'.format(title,tickers[i],last_values[i]*factors[i],factors[i], last_values[i]))

    return leg

def plot_data(start,end, tickers):
    # Set start and end
    start = dt.datetime.strptime(start,'%Y/%m/%d')
    end = dt.datetime.strptime(end,'%Y/%m/%d')

    # Load the data
    main_df = load_data(tickers)

    # Get subset of time
    df = get_timeframe(main_df,start,end)

    # normalize
    df, factors = normalize(df)

    # Sort
    sort_order = np.argsort(df.loc[df.last_valid_index()].to_numpy()*-1)
    factors = np.array(factors)[sort_order]
    df = df.sort_values(df.last_valid_index(), ascending = False, axis=1)

    # Get last values
    last_values = df.iloc[-1]

    # Get legend
    leg = create_legend(list(df),last_values,factors)

    # Register plotting converter
    pd.plotting.register_matplotlib_converters()

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
        if not args.tickers:
            tickers = get_tickers()
        else:
            tickers = args.tickers
        plot_data(start = args.start, end = args.end, tickers=tickers)
    if args.func == 'update':
        update_data()
    if args.func == 'download':
        download_data(start = args.start, end = args.end, force = args.force)
