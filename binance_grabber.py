import datetime as dt
import logging

import environs
import numpy as np
import pandas as pd
from binance.client import Client as BinanceClient
from tqdm import tqdm

env = environs.Env()
env.read_env()


BINANCE_API_KEY = env.str('BINANCE_API_KEY')
BINANCE_API_SECRET = env.str('BINANCE_API_SECRET')


client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)


def get_market_bars(symbol: str, interval: str, start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame:
    headers = (
        'Open time',
        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
        'Close time',
        'Quote volume',
        'Trades',
        'Buy base volume',
        'Buy quote volume',
        'Ignore',
    )
    start_ts = str(int(start_dt.timestamp()))
    end_ts = str(int(end_dt.timestamp()))

    data_generator = client.get_historical_klines_generator(symbol, interval, start_ts, end_ts)
    df = pd.DataFrame(tqdm(data_generator), columns=headers)

    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

    df['Open'] = (df['Open'].astype(np.float32) * 100).astype(np.int32)
    df['High'] = (df['High'].astype(np.float32) * 100).astype(np.int32)
    df['Low'] = (df['Low'].astype(np.float32) * 100).astype(np.int32)
    df['Close'] = (df['Close'].astype(np.float32) * 100).astype(np.int32)

    return (
        df
        .drop(['Quote volume', 'Buy base volume', 'Buy quote volume', 'Ignore'], axis=1)
        .sort_values('Open time')
    )
