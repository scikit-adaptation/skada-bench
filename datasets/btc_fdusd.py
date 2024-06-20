from benchopt import BaseDataset, safe_import_context
with safe_import_context() as import_ctx:
    import scipy
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from skada.utils import source_target_merge
    import pandas as pd
    import warnings
    from pandas.errors import SettingWithCopyWarning, PerformanceWarning

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=PerformanceWarning)

# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "BTCFDUSD"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'source_target': [
            ('BTCUSDT', 'BTCFDUSD')
        ],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        datasets_path = '../../stock_trading/data/'

        X_source, y_source, df_source = get_train_X_Y(variations=True)
        X_target, y_target, df_target = get_test_X_Y(variations=True)

        # Drop columns with inf values
        for df in [X_source, X_target]:
            columns_with_inf = df.columns[np.where(np.isinf(df))[1]]
            if not columns_with_inf.empty:
                X_source.drop(columns_with_inf, axis=1, inplace=True)
                X_target.drop(columns_with_inf, axis=1, inplace=True)

        # source = self.source_target[0]
        # target = self.source_target[1]

        X_source = X_source.values
        X_target = X_target.values

        X, y, sample_domain = source_target_merge(
            X_source, X_target, y_source, y_target)

        return dict(
            X=X,
            y=y,
            sample_domain=sample_domain,
        )


def get_train_X_Y(variations = False):
    df = pd.read_csv('../../stock_trading/data/binance_bitcoin_data.txt', sep=';')
    return prepocess_data(df, variations=variations)

def get_test_X_Y(variations = False):
    df = pd.read_csv('../../stock_trading/CryptoBot/test_models/test_data/btcfdusd_01_03_24-09_04_24.csv', sep=';')
    return prepocess_data(df, variations=variations)


def prepocess_data(df, variations = False):

    df['open_timestamp'] = pd.to_datetime(df['open_timestamp'], unit='ms')
    df['close_timestamp'] = pd.to_datetime(df['close_timestamp'], unit='ms')

    time_list = ['10min', '20min', '30min', '1h', '20h', '50h']
    #time_list = ['10min', '20min', '30min']

    columns_name = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'quote_volume', 'trade_count', 'taker_buy_volume', 'taker_buy_quote_volume']

    for timeee in time_list:
        for feature_namee in columns_name:
            name = "_".join([feature_namee, timeee, "median"])
            df[name] = df[['open_timestamp', feature_namee]].rolling(timeee, center=False, on='open_timestamp').median()[feature_namee]
            df[name] = df[name].ffill().bfill()
            df[name] = df[name].astype(df[feature_namee].dtype)

            name = "_".join([feature_namee, timeee, "min"])
            df[name] = df[['open_timestamp', feature_namee]].rolling(timeee, center=False, on='open_timestamp').min()[feature_namee]
            df[name] = df[name].ffill().bfill()
            df[name] = df[name].astype(df[feature_namee].dtype)

            name = "_".join([feature_namee, timeee, "mean"])
            df[name] = df[['open_timestamp', feature_namee]].rolling(timeee, center=False, on='open_timestamp').mean()[feature_namee]
            df[name] = df[name].ffill().bfill()
            df[name] = df[name].astype(df[feature_namee].dtype)

            name = "_".join([feature_namee, timeee, "std"])
            df[name] = df[['open_timestamp', feature_namee]].rolling(timeee, center=False, on='open_timestamp').std()[feature_namee]
            df[name] = df[name].ffill().bfill()
            df[name] = df[name].astype(df[feature_namee].dtype)

            name = "_".join([feature_namee, timeee, "max"])
            df[name] = df[['open_timestamp', feature_namee]].rolling(timeee, center=False, on='open_timestamp').max()[feature_namee]
            df[name] = df[name].ffill().bfill()
            df[name] = df[name].astype(df[feature_namee].dtype)

    features_to_keep = ['close_price', 'high_price', 'high_price_30min_max', 'low_price_30min_min', 'high_price_20min_max']

    df = df.drop(columns=['pair'])
    df = df.dropna()

    features_to_keep = [feature for feature in df.columns if feature not in ['open_timestamp', 'close_timestamp', 'Y', 'pair']]
    if variations:
        for feature in features_to_keep :
            df[feature + '_variation_10min'] = (df[feature] - df[feature].shift(1)) / df[feature].shift(1)
            df[feature + '_variation_20min'] = (df[feature] - df[feature].shift(2)) / df[feature].shift(3)
            df[feature + '_variation_30min'] = (df[feature] - df[feature].shift(4)) / df[feature].shift(5)
            df[feature + '_variation_1h'] = (df[feature] - df[feature].shift(12)) / df[feature].shift(11)
            #df[feature] = (df[feature] - df[feature].shift(1)) / df[feature].shift(1)

        df = df.dropna()

    #features_to_keep = features_to_keep + [feature + '_variation' for feature in features_to_keep]
    #df['Y'] = df['close_price_variation'].shift(-1)
    df['Y'] = df['close_price'].shift(-1)
    df = df.dropna()
    #df['Y'] = (df['Y'] > 0).astype(int)
    df['Y'] = (df['Y'] > df['close_price']).astype(int) 

    #df['year'] = pd.DatetimeIndex(df['open_timestamp']).year
    df['month'] = pd.DatetimeIndex(df['open_timestamp']).month
    df['day'] = pd.DatetimeIndex(df['open_timestamp']).day
    df['hour'] = pd.DatetimeIndex(df['open_timestamp']).hour
    df['minute'] = pd.DatetimeIndex(df['open_timestamp']).minute

    df = df.drop(columns=['open_timestamp', 'close_timestamp'])
    #features_to_keep = [feature for feature in df.columns if feature not in ['open_timestamp', 'close_timestamp', 'Y', 'pair']]
    features_to_keep = ['close_price_variation_10min',
    'minute',
    'open_price_30min_std_variation_1h',
    'hour',
    'close_price_20min_std_variation_10min',
    'open_price_30min_std_variation_20min',
    'taker_buy_quote_volume_10min_std_variation_10min',
    'close_price_30min_std_variation_10min',
    'open_price_10min_std',
    'trade_count_10min_std',
    'taker_buy_quote_volume_10min_std_variation_30min',
    'close_price_1h_std_variation_1h',
    'high_price_10min_std',
    'high_price_variation_10min',
    'close_price_10min_max_variation_10min',
    'taker_buy_quote_volume_10min_std_variation_20min',
    'taker_buy_quote_volume_10min_std_variation_1h',
    'trade_count_20h_median_variation_10min',
    'month',
    'open_price_variation_10min',
    'low_price_variation_10min',
    'trade_count_30min_std_variation_10min',
    'high_price_1h_min_variation_1h',
    'close_price_1h_std_variation_10min',
    'taker_buy_volume_50h_median_variation_1h',
    'low_price_10min_std',
    'close_price_10min_std',
    'open_price_20min_std',
    'taker_buy_quote_volume_50h_max',
    'close_price_10min_max_variation_20min']

    df = df.dropna()

    # features_to_keep = ['close_price_variation_5min', 'close_price_variation_10min', 'close_price_variation_30min',
    #                 'low_price_10min_std', 'minute', 'high_price_variation_5min', 'trade_count_10min_std',
    #                 'high_price_variation_10min', 'close_price_20min_max', 'hour', 'high_price', 'close_price']
    
    return df[features_to_keep], df['Y'].values.reshape(-1, 1).flatten(), df
