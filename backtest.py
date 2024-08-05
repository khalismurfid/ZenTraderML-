# backtest.py

from stratestic.backtesting import IterativeBacktester

def run_backtest(strategy_class, data, symbol, **strategy_kwargs):
    """Run backtest for a given strategy."""
    strategy = strategy_class(data=data, **strategy_kwargs)
    ite = IterativeBacktester(strategy, symbol=symbol)
    ite.load_data(data)
    ite.run()
