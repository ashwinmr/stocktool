import argparse
import datetime as dt
from posixpath import dirname
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import os
import pickle
from mpldatacursor import datacursor

# Global variables

TICKER_INFO_FILE = 'ticker_info.csv'
TICKER_DATA_PATH = 'sec_dfs'
# A mapping from legend to line for clickable legend
LegToLine = dict()


class Securities:
    def __init__(self):
        self.ticker_info = self.load_ticker_info()

    def load_ticker_info(self):
        """ Load the ticker info
        """
        dirname = os.path.dirname(__file__)
        ticker_info_path = os.path.join(dirname, TICKER_INFO_FILE)
        return pd.read_csv(ticker_info_path, index_col='Ticker')

    def get_ticker_data_path(self, ticker):
        """ Get the file path for ticker data
        """
        dirname = os.path.dirname(__file__)
        ticker_data_path = os.path.join(
            dirname, TICKER_DATA_PATH, '{}.csv'.format(ticker))
        return ticker_data_path

    def get_ticker_data(self, ticker):
        """ Get the data for a ticker
        """
        try:
            # print('Compiling {}'.format(ticker))
            ticker_data_path = self.get_ticker_data_path(ticker)
            df = pd.read_csv(ticker_data_path,
                             index_col='Date', parse_dates=['Date'])
            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
            return df
        except Exception as e:
            print('Failed to compile {}:\n\t{}'.format(ticker, e))
            return None

    def get_all_tickers(self):
        """ Get all tickers
        """
        return self.ticker_info.index.tolist()

    def get_tickers_by_group(self, group):
        """ Get tickers by group
        """
        df = self.ticker_info
        return df.loc[df['Group'] == group].index.tolist()

    def get_tickers_by_country(self, country):
        """ Get tickers by country
        """
        df = self.ticker_info
        return df.loc[df['Country'] == country].index.tolist()

    def get_title(self, ticker):
        """ Get title for ticker
        """
        title = self.ticker_info.loc[ticker, 'Title']
        return title

    def load_data(self, tickers):
        """ Load data for tickers
        """
        main_df = pd.DataFrame()
        for ticker in tickers:
            df = self.get_ticker_data(ticker)
            if df is not None:
                if main_df.empty:
                    main_df = df
                else:
                    main_df = main_df.join(df, how='outer')
        return main_df

    def update_ticker_data(self, tickers):
        """ Update the security data to the current date
        """
        for ticker in tickers:
            try:
                print('Updating {}'.format(ticker))

                # Load old data
                df_old = pd.read_csv(self.get_ticker_data_path(
                    ticker), index_col='Date', parse_dates=['Date'])

                # Get last date
                last_date = df_old.index[-1]

                # Download new data
                start = last_date
                end = dt.datetime.today()
                df_new = web.DataReader(ticker, 'yahoo', start, end)

                # Append
                df = df_old.append(df_new)

                # Remove duplicates
                # Dropping duplicates using index doesn't work
                df = df.reset_index().drop_duplicates(subset='Date').set_index('Date')

                # Save
                df.to_csv(self.get_ticker_data_path(ticker))

            except Exception as e:
                print('Failed to update {}:\n\t{}'.format(ticker, e))

    def download_ticker_data(self, start, end, tickers, force=False):
        """ Download data from yahoo for provided tickers
        """
        if not os.path.exists(TICKER_DATA_PATH):
            os.makedirs(TICKER_DATA_PATH)

        for ticker in tickers:
            if not os.path.exists(self.get_ticker_data_path(ticker)) or force:
                try:
                    print('Downloading {}'.format(ticker))
                    df = web.DataReader(ticker, 'yahoo', start, end)
                    df.to_csv(self.get_ticker_data_path(ticker))
                except Exception as e:
                    print('Failed to download {}:\n\t{}'.format(ticker, e))
            else:
                print('Already have {}'.format(ticker))

    def process_ticker_data(self, start, end, tickers, resample_string=None):
        """ Get subset of resampled data
        """
        # Set start and end
        start = dt.datetime.strptime(start, '%Y/%m/%d')
        end = dt.datetime.strptime(end, '%Y/%m/%d')

        # Load the data
        df = self.load_data(tickers)

        # Get subset of time
        df = get_timeframe(df, start, end)

        # Resample
        if resample_string is not None:
            df = df.resample(resample_string).mean()

        return df

    def output_ticker_data(self, start, end, resample_string, tickers, file_path):
        """ Output data to csv
        """
        df = self.process_ticker_data(
            start=start, end=end, tickers=tickers, resample_string=resample_string)
        print(df)
        df.to_csv(file_path)

    def js_output(self, start, end, resample_string, tickers):
        """ Output data for js
        """
        df = self.process_ticker_data(
            start=start, end=end, tickers=tickers, resample_string=resample_string)

        # Output to temp csv
        temp_csv_path = os.path.join(get_temp_dir(),'js_output.csv')
        df.to_csv(temp_csv_path)
        print(df.to_csv())

    def plot_data(self, start, end, tickers):
        # Set start and end
        start = dt.datetime.strptime(start, '%Y/%m/%d')
        end = dt.datetime.strptime(end, '%Y/%m/%d')

        # Load the data
        main_df = self.load_data(tickers)

        # Get subset of time
        df = get_timeframe(main_df, start, end)

        # normalize
        df, factors = normalize(df)

        # Get last non null indexes
        idx = df.apply(pd.Series.last_valid_index)

        # Get last non null values
        last_values = []
        for col in df:
            last_value = df[col].loc[idx[col]]
            last_values.append(last_value)

        # Sort
        sort_order = np.argsort(np.array(last_values)*-1)
        factors = np.array(factors)[sort_order]
        last_values = np.array(last_values)[sort_order]
        df = reorder_cols(df, sort_order)
        tickers = list(df)

        # Get legend
        leg = []
        titles = [self.get_title(ticker) for ticker in tickers]
        for ticker, title, factor, last_value in zip(tickers, titles, factors, last_values):
            leg.append('{} ({}): ({:.0f}/{:.0f}) {:.2f}'.format(title,
                       ticker, last_value*factor, factor, last_value))

        # Register plotting converter
        pd.plotting.register_matplotlib_converters()

        # Plot
        plt.plot(df)
        plt.xlabel('date')
        plt.ylabel('gain')
        plt.grid('True')
        plt.legend(leg)

        # Shrink current axis by 20%
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        leg = ax.legend(leg, loc='center left',
                        bbox_to_anchor=(1, 0.5), prop={'size': 8})

        # Make the legend clickable
        for legline, origline in zip(leg.get_lines(), ax.get_lines()):
            legline.set_picker(5)  # 5pt tolerance
            LegToLine[legline] = origline

        # Add data cursors
        datacursor(display='multiple', draggable=True)

        # Handle pick events
        plt.gcf().canvas.mpl_connect('pick_event', onpick)

        plt.show()

def get_temp_dir():
    # Get 
    program_dir = os.path.dirname(os.path.realpath(__file__))
    temp_dir = os.path.join(program_dir,'temp')
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

def normalize(df_in):

    # Store normalization factors
    factors = []

    # Get first non null indexes
    idx = df_in.apply(pd.Series.first_valid_index)

    # Copy
    df = df_in.copy()

    # Normalize each column
    for col in df:
        factor = df[col].loc[idx[col]]
        factors.append(factor)
        df[col] = df[col]/factor

    return df, factors


def reorder_cols(df, order):
    """ Reorder columns of a dataframe
    """

    # Get the list of columns and reorder them
    cols = df.columns.tolist()
    cols = np.array(cols)[order]

    return df[cols]


def get_timeframe(df, start, end):
    """ Get timeframe section from df
    """

    # Create mask between time
    mask = (df.index > start) & (df.index < end)

    # Set output
    df_out = df.loc[mask].copy()

    # Remove empty columns
    df_out.dropna(axis='columns', how='all', inplace=True)

    return df_out

def onpick(event):
    """ Handle pick event on the legend
    """

    # on the click event, find the orig line corresponding to the legend line, and toggle visibility
    legline = event.artist
    if legline in LegToLine:
        origline = LegToLine[legline]
    else:
        return
    vis = not origline.get_visible()
    origline.set_visible(vis)
    # Change the alpha on the line in the legend so we can see what lines have been toggled
    if vis:
        legline.set_alpha(1.0)
    else:
        legline.set_alpha(0.2)

    # Redraw the figure
    plt.gcf().canvas.draw()


def parse_args():
    """ Parse arguments for program
    """
    parser = argparse.ArgumentParser(description="Program for analyzing stock")
    subparsers = parser.add_subparsers(dest='sub_cmd', required=True)

    plot_parser = subparsers.add_parser('plot', help='plot the stock data')
    plot_parser.add_argument(
        '-s', '--start', default='2021/1/1', help="start date in the format YYYY/mm/dd")
    plot_parser.add_argument('-e', '--end', default=dt.datetime.today().strftime(
        "%Y/%m/%d"), help="end date in the format YYYY/mm/dd")
    plot_parser.add_argument(
        '-t', '--tickers', nargs='*', help='List of tickers')
    plot_parser.add_argument('-g', '--group', help='Group')
    plot_parser.add_argument('-c', '--country', help='Country')
    plot_parser.set_defaults(func='plot')

    output_parser = subparsers.add_parser(
        'output', help='output the stock data')
    output_parser.add_argument(
        '-s', '--start', default='2018/1/1', help="start date in the format YYYY/mm/dd")
    output_parser.add_argument('-e', '--end', default=dt.datetime.today().strftime(
        "%Y/%m/%d"), help="end date in the format YYYY/mm/dd")
    output_parser.add_argument(
        '-t', '--tickers', nargs='*', help='List of tickers')
    output_parser.add_argument(
        '-r', '--resample_string', help='Resample string')
    output_parser.add_argument(
        '-f', '--file_path', default='temp/output.csv', help='Output file path')
    output_parser.add_argument('-g', '--group', help='Group')
    output_parser.add_argument('-c', '--country', help='Country')
    output_parser.set_defaults(func='output')

    update_parser = subparsers.add_parser(
        'update', help='update the stock data')
    update_parser.add_argument(
        '-t', '--tickers', nargs='*', help='List of tickers')
    update_parser.add_argument('-g', '--group', help='Group')
    update_parser.add_argument('-c', '--country', help='Country')
    update_parser.set_defaults(func='update')

    download_parser = subparsers.add_parser(
        'download', help='download the stock data')
    download_parser.add_argument(
        '-s', '--start', default='2000/1/1', help="start date in the format YYYY/mm/dd")
    download_parser.add_argument('-e', '--end', default=dt.datetime.today(
    ).strftime("%Y/%m/%d"), help="end date in the format YYYY/mm/dd")
    download_parser.add_argument(
        '-f', '--force', action='store_true', help="force download")
    download_parser.add_argument(
        '-t', '--tickers', nargs='*', help='List of tickers')
    download_parser.add_argument('-g', '--group', help='Group')
    download_parser.add_argument('-c', '--country', help='Country')
    download_parser.set_defaults(func='download')

    js_output_parser = subparsers.add_parser(
        'js_output', help='output for use in js')
    js_output_parser.add_argument(
        '-s', '--start', default='2018/1/1', help="start date in the format YYYY/mm/dd")
    js_output_parser.add_argument('-e', '--end', default=dt.datetime.today(
    ).strftime("%Y/%m/%d"), help="end date in the format YYYY/mm/dd")
    js_output_parser.add_argument(
        '-t', '--tickers', nargs='*', help='List of tickers')
    js_output_parser.add_argument(
        '-r', '--resample_string', help='Resample string')
    js_output_parser.add_argument('-g', '--group', help='Group')
    js_output_parser.add_argument('-c', '--country', help='Country')
    js_output_parser.set_defaults(func='js_output')

    return parser.parse_args()

if __name__ == "__main__":

    # Initialize securities object
    secs = Securities()

    args = parse_args()

    # Select tickers
    if args.group:
        tickers = secs.get_tickers_by_group(args.group)
    elif args.country:
        tickers = secs.get_tickers_by_country(args.country)
    elif args.tickers:
        tickers = args.tickers
    else:
        tickers = secs.get_all_tickers()

    if args.func == 'plot':
        secs.plot_data(start=args.start, end=args.end, tickers=tickers)
    if args.func == 'update':
        secs.update_ticker_data(tickers=tickers)
    if args.func == 'download':
        secs.download_ticker_data(
            start=args.start, end=args.end, tickers=tickers, force=args.force)
    if args.func == 'output':
        if not args.resample_string:
            resample_string = None
        else:
            resample_string = args.resample_string
        secs.output_ticker_data(start=args.start, end=args.end,
                                resample_string=resample_string, tickers=tickers, file_path=args.file_path)
    if args.func == 'js_output':
        if not args.resample_string:
            resample_string = None
        else:
            resample_string = args.resample_string
        secs.js_output(start=args.start, end=args.end,
                       resample_string=resample_string, tickers=tickers)
