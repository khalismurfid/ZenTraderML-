from collections import OrderedDict
import numpy as np
import pandas as pd
from ta.trend import MACD, EMAIndicator, ADXIndicator  # ADXIndicator is under trend
from ta.momentum import StochasticOscillator, RSIIndicator
from stratestic.backtesting.helpers.evaluation import SIDE
from stratestic.strategies._mixin import StrategyMixin

class LongOnlyMACD(MACD, StrategyMixin):
    """
    Long-only MACD strategy that uses 3 different parameters to calculate the side.
    It calculates the difference between the slow and the fast exponential moving averages,
    and compares that with the signal window to get the signal. It goes long when the signal
    moving average goes above the difference of the fast and slow.

    Parameters
    ----------
    window_slow : int
        Slow moving average window, by default 26
    window_fast : int
        Fast moving average window, by default 12
    window_sign : int
        Signal moving average window, by default 9
    data : pd.DataFrame, optional
        Data to use, by default None
    **kwargs
        Keyword arguments to pass to parent class

    Attributes
    ----------
    params : collections.OrderedDict
        Parameters of the strategy
    _close : pd.Series
        Close prices of the data
    """

    def __init__(self, window_slow: int, window_fast: int, window_sign: int, data: pd.DataFrame = None, **kwargs):
        MACD.__init__(self, pd.Series(dtype='float64'), window_slow, window_fast, window_sign)
        StrategyMixin.__init__(self, data, **kwargs)

        self._close = pd.Series(dtype='float64')

        self.params = OrderedDict(
            window_slow=lambda x: int(x),
            window_fast=lambda x: int(x),
            window_sign=lambda x: int(x)
        )

    def __repr__(self) -> str:
        return "{}(symbol = {}, fast = {}, slow = {}, signal = {})".format(
            self.__class__.__name__, self.symbol, self._window_fast, self._window_slow, self._window_sign
        )

    def update_data(self, data) -> None:
        data = super().update_data(data)
        self._close = data[self._close_col]
        self._run()
        data["macd_diff"] = self.macd_diff()
        return data

    def calculate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        # Long-only: Only take long positions where MACD difference is positive
        data[SIDE] = np.where(data["macd_diff"] > 0, 1, 0)
        return data

    def get_signal(self, row=None):
        if row is None:
            row = self.data.iloc[-1]

        return 1 if row["macd_diff"] > 0 else 0

class LongOnlyMomentum(StrategyMixin):
    """
    Long-only Momentum strategy that calculates the rolling average return over a specified window of time.
    It generates a long signal when the rolling average return is positive.

    Parameters
    ----------
    window : int
        Rolling window for computing the momentum.
    data : pd.DataFrame
        Input data with OHLC columns.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    _window : int
        Rolling window for computing the momentum.
    params : OrderedDict
        Dictionary of hyperparameters for the strategy.

    Methods
    -------
    update_data()
        Retrieves and prepares the data.
    calculate_positions(data)
        Calculates the positions of the strategy.
    get_signal(row=None)
        Returns the trading signal for a given row of data.
    """

    def __init__(self,  window: int, data=None, **kwargs):
        self._window = window

        StrategyMixin.__init__(self, data, **kwargs)

        self.params = OrderedDict(window=lambda x: int(x))

    def __repr__(self):
        return "{}(symbol = {}, window = {})".format(self.__class__.__name__, self.symbol, self._window)

    def update_data(self, data):
        data = super().update_data(data)

        # Calculate rolling returns
        data["rolling_returns"] = data[self._returns_col].rolling(self._window, min_periods=1).mean()

        return self.calculate_positions(data)

    def calculate_positions(self, data):
        # Long-only: Only take long positions where rolling returns are positive
        data[SIDE] = np.where(data["rolling_returns"] > 0, 1, 0)
        return data

    def get_signal(self, row=None):
        if row is None:
            row = self.data.iloc[-1]

        return 1 if row["rolling_returns"] > 0 else 0

class StochasticOscillatorStrategy(StrategyMixin):
    """Stochastic Oscillator-based long-only strategy."""
    def __init__(self, k_window=14, smooth_k=3, smooth_d=3, threshold=20, data=None, **kwargs):
        self._k_window = k_window
        self._smooth_k = smooth_k
        self._smooth_d = smooth_d
        self._threshold = threshold

        StrategyMixin.__init__(self, data, **kwargs)

        self.params = OrderedDict(
            k_window=lambda x: int(x),
            smooth_k=lambda x: int(x),
            smooth_d=lambda x: int(x),
            threshold=lambda x: int(x)
        )

    def __repr__(self):
        return "{}(symbol = {}, k_window = {}, smooth_k = {}, smooth_d = {}, threshold = {})".format(
            self.__class__.__name__, self.symbol, self._k_window, self._smooth_k, self._smooth_d, self._threshold)

    def update_data(self, data):
        """Calculate the Stochastic Oscillator (%K and %D) for the given data."""
        high = data['high']
        low = data['low']
        close = data['close']

        lowest_low = low.rolling(window=self._k_window).min()
        highest_high = high.rolling(window=self._k_window).max()

        data['%k'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        data['%k'] = data['%k'].rolling(window=self._smooth_k).mean()  # Smooth %K
        data['%d'] = data['%k'].rolling(window=self._smooth_d).mean()  # Smooth %D

        return self.calculate_positions(data)

    def calculate_positions(self, data):
        """Calculate positions based on strategy rules."""
        data[SIDE] = np.where((data['%k'] < self._threshold) & (data['%k'] > data['%d']), 1, 0)
        return data

    def get_signal(self, row=None):
        if row is None:
            row = self.data.iloc[-1]
        return int(row[SIDE])

class StochasticEMAStrategy(StrategyMixin):
    """Stochastic Oscillator and EMA Strategy with ADX"""
    def __init__(self, k_length, k_smoothing, d_smoothing, adx_length, adx_smoothing, adx_threshold, rsi_length, data=None, **kwargs):
        self._k_length = k_length
        self._k_smoothing = k_smoothing
        self._d_smoothing = d_smoothing
        self._adx_length = adx_length
        self._adx_smoothing = adx_smoothing
        self._adx_threshold = adx_threshold
        self._rsi_length = rsi_length

        StrategyMixin.__init__(self, data, **kwargs)

        self.params = OrderedDict(
            k_length=lambda x: int(x),
            k_smoothing=lambda x: int(x),
            d_smoothing=lambda x: int(x),
            adx_length=lambda x: int(x),
            adx_smoothing=lambda x: int(x),
            adx_threshold=lambda x: float(x),
            rsi_length=lambda x: int(x)
        )

    def __repr__(self):
        return "{}(symbol = {}, k_length = {}, k_smoothing = {}, d_smoothing = {}, adx_length = {}, adx_smoothing = {}, adx_threshold = {}, rsi_length = {})".format(
            self.__class__.__name__, self.symbol, self._k_length, self._k_smoothing, self._d_smoothing, self._adx_length, self._adx_smoothing, self._adx_threshold, self._rsi_length)

    def update_data(self, data):
        """Calculate the Stochastic Oscillator, EMAs, ADX, and RSI for the given data."""
        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate EMAs
        ema20 = EMAIndicator(close, 20).ema_indicator()
        ema50 = EMAIndicator(close, 50).ema_indicator()
        ema100 = EMAIndicator(close, 100).ema_indicator()
        ema200 = EMAIndicator(close, 200).ema_indicator()
        ema400 = EMAIndicator(close, 400).ema_indicator()

        # Calculate Stochastic Oscillator
        stoch = StochasticOscillator(high=high, low=low, close=close, window=self._k_length, smooth_window=self._k_smoothing)
        data['%k'] = stoch.stoch()
        data['%d'] = data['%k'].rolling(window=self._d_smoothing).mean()  # Smooth %D

        # Calculate ADX
        adx = ADXIndicator(high=high, low=low, close=close, window=self._adx_length).adx()
        data['adx'] = adx

        # Calculate RSI
        rsi = RSIIndicator(close=close, window=self._rsi_length).rsi()
        data['rsi'] = rsi

        # Determine highest EMA
        data['highest_ema'] = np.maximum.reduce([ema20, ema50, ema100, ema200, ema400])

        return self.calculate_positions(data)

    def calculate_positions(self, data):
        """Calculate positions based on strategy rules."""
        # Ensure the SIDE column is initialized to zero
        data[SIDE] = data.get(SIDE, 0)
    
        buy_condition = (data['%k'] < 2) & (data[SIDE].shift(1).fillna(0) == 0)
        sell_condition = (data['close'] >= data['highest_ema']) & (data[SIDE].shift(1).fillna(0) == 1)
    
        data[SIDE] = np.where(buy_condition, 1, np.where(sell_condition, 0, data[SIDE].shift(1).fillna(0)))
        return data


    def get_signal(self, row=None):
        if row is None:
            row = self.data.iloc[-1]
        return int(row[SIDE])
