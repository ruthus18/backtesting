import datetime as dt
import logging
import typing as t
from decimal import Decimal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import binance_grabber

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Indicator:

    def get_trade_signals(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """Интерфейс для получения DataFrame с сигналами на покупку и продажу.

        Конечный DataFrame должен содержать столбцы с индексами:
            * Time - время совершения сделки
            * Signal - сигнал на покупку (1) или продажу (-1)
        """
        raise NotImplementedError


class MACDIndicator(Indicator):

    def __init__(self, ema_long_period: int = 26, ema_short_period: int = 12, signal_ema_period: int = 9):
        self._ema_long_period = ema_long_period
        self._ema_short_period = ema_short_period
        self._signal_ema_period = signal_ema_period

        self.ema_long = None
        self.ema_short = None
        self.macd = None
        self.signal_ema = None
        self.macd_histogram = None

    def _calc_indicator(self, bars_df: pd.DataFrame):
        assert 'Close' in bars_df, 'Bars data has wrong columns'

        self.ema_long = bars_df['Close'].ewm(span=26, adjust=False).mean()
        self.ema_short = bars_df['Close'].ewm(span=12, adjust=False).mean()
        self.macd = self.ema_short - self.ema_long

        self.signal_ema = self.macd.ewm(span=9, adjust=False).mean()
        self.macd_histogram = self.macd - self.signal_ema

    def get_trade_signals(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        self._calc_indicator(bars_df)

        histogram_signs = np.sign(self.macd_histogram)

        # на основе знака гистограммы генерируем сигналы на покупку и продажу
        # -1 - сигнал продажи, 1 - сигнал покупки, 0 - держим позицию
        trade_signals = np.sign(np.concatenate(([0],np.diff(histogram_signs)))).astype(np.short)

        trades = pd.DataFrame({
            'Time': bars_df['Open time'],
            'Signal': trade_signals
        })
        trades = trades[trades['Signal'] != 0].reset_index(drop=True)

        # Пропускаем первый сигнал покупки-продажи, т.к. он ложный из-за отсутствия данных
        # о смене знака в начале гистограммы
        if trades['Signal'][0] == 1:
            trades = trades[2:]

        elif trades['Signal'][0] == -1:
            trades = trades[1:]

        return trades


class BinanceBacktester:
    POSITION_COMISSION = Decimal(0.0015)

    def __init__(self, bars_data: pd.DataFrame, indicator: Indicator):
        self.bars_data = bars_data
        self.indicator = indicator

        self._trade_positions = None

    @classmethod
    def _calc_profit(cls, row: pd.Series) -> Decimal:
        return Decimal(row['Price sell']) / Decimal(row['Price buy']) - 1 - cls.POSITION_COMISSION

    @staticmethod
    def _calc_result_profit(row: pd.Series) -> Decimal:
        result = Decimal(1)
        for profit in row.values:
            result *= (1 + Decimal(profit))
        
        return result - 1

    @staticmethod
    def _calc_equity(profits: pd.Series) -> t.List[Decimal]:
        equities = []
        last_equity = Decimal(1)
        for profit in profits:
            equity = last_equity * (1 + Decimal(profit))
            equities.append(equity)
            last_equity = equity
            
        return equities

    def _get_trade_positions(self) -> pd.DataFrame:
        """Получить DataFrame с данными о совершенных сделках на основе сигналов индикатора

        Returns (DataFrame columns):
            * Time buy - время открытия позиции
            * Price buy - цена открытия позиции
            * Time close - время закрытия позиции
            * Price close - время закрытия позиции
        """
        data_cols = self.bars_data.columns
        assert 'Open time' in data_cols and 'Open' in data_cols, 'Bars data has wrong columns'

        # 1. Get trades info

        trade_singals = self.indicator.get_trade_signals(self.bars_data)
        trades = pd.DataFrame({
            'Time': self.bars_data['Open time'],
            # Совершаем сделку по цене открытия следующей свечи
            'Price': self.bars_data['Open'].shift(-1)
        }).merge(
            trade_singals, on=['Time']
        )
        trades['Price'] = trades['Price'].astype(np.int32)

        # 2. Get trade positions

        buy_trades = trades[trades['Signal'] == 1].reset_index(drop=True)
        sell_trades = trades[trades['Signal'] == -1].reset_index(drop=True)

        # Попарно соединяем сделки покупки/продажи в позицию
        trade_positions = (
            buy_trades.merge(sell_trades, suffixes=(' buy', ' sell'), left_index=True, right_index=True)
            .reset_index(drop=True)
            .drop(['Signal buy', 'Signal sell'], axis=1)
        )

        trade_positions['Profit'] = trade_positions.apply(self._calc_profit, axis=1)
        trade_positions['Equity'] = self._calc_equity(trade_positions['Profit'].values)

        return trade_positions

    @property
    def trade_positions(self):
        if self._trade_positions is None:
            self._trade_positions = self._get_trade_positions()

        return self._trade_positions

    def export_trade_positions(self):
        export_df = self.trade_positions.drop(['Profit', 'Equity'], axis=1)
        export_df['Price buy'] = export_df['Price buy'] / 100
        export_df['Price sell'] = export_df['Price sell'] / 100
        
        export_df.to_csv('trades.csv', index=False)

    def export_equity_graph(self):
        equity_df = self.trade_positions[['Time sell', 'Equity']].shift()
        equity_df.iloc[0] = (self.bars_data['Open time'][0], 1)

        equity_graph = go.Figure(data=[
            go.Scatter(
                name='Equity', x=equity_df['Time sell'], y=equity_df['Equity'], line=dict(color='#fe8019', width=1)
            ),
        ])
        equity_graph.update_layout(template='plotly_white')
        equity_graph.write_image("equity.png")

    def run_test(self):
        profits = self.trade_positions['Profit']

        pos_profits = profits[profits >= 0]
        neg_profits = profits[profits < 0]

        mean_pos_profits = pos_profits.mean() 
        mean_neg_profits = neg_profits.mean()

        trades_count = len(self.trade_positions)
        pos_trades_count = len(pos_profits)
        neg_trades_count = len(neg_profits)

        relative_profit_trades = Decimal(pos_trades_count) / Decimal(trades_count)
        relative_loss_trades = Decimal(neg_trades_count) / Decimal(trades_count)

        result_profit = self._calc_result_profit(profits)

        report = f'''
        Indicator report
        ----------------------------------------------------
        Net mean of pos trades: {mean_pos_profits}
        Net mean of neg trades: {mean_neg_profits}

        Count of trades: {trades_count}
        Count of pos trades: {pos_trades_count}
        Count of neg trades: {neg_trades_count}

        Relative profit trades: {relative_profit_trades}
        Relative loss trades: {relative_loss_trades}

        Result profit: {result_profit:.16f}
        '''
        logger.info(report)


def main():
    symbol = 'BTCUSDT'
    interval = '1h'
    start_dt = dt.datetime(2018, 1, 1)
    end_dt = dt.datetime(2021, 1, 1)

    logging.info(
        'Grabbing Binance data for %s (%s - %s , interval = %s)...',
        symbol, start_dt.date(), end_dt.date(), interval
    )

    bars_data = binance_grabber.get_market_bars(symbol, interval, start_dt, end_dt)

    logging.info('Run backtesting...')
    backtester = BinanceBacktester(bars_data, MACDIndicator())

    backtester.export_trade_positions()
    logger.info('Trades data exported to "trades.csv"')

    backtester.export_equity_graph()
    logger.info('Equity graph exported to "equity.png"')

    backtester.run_test()


if __name__ == '__main__':
    main()    
