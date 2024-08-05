# backtest.py

from stratestic.backtesting.iterative import IterativeBacktester as OriginalIterativeBacktester
from stratestic.backtesting.helpers.evaluation import CUM_SUM_STRATEGY, CUM_SUM_STRATEGY_TC, BUY_AND_HOLD

class IterativeBacktesterWithResults(OriginalIterativeBacktester):
    def run(self, print_results=False):
        """Runs the backtest and returns performance metrics."""
        # Call the original run method and optionally print results
        perf, outperf, results = self._test_strategy(print_results=print_results, plot_results=print_results)
        return perf, outperf, results

def run_backtest(strategy_class, data, symbol, print_results=False, **strategy_kwargs):
    """Run backtest for a given strategy and return result metrics."""
    # Initialize the strategy
    strategy = strategy_class(data=data, **strategy_kwargs)
    
    # Set up the iterative backtester with the strategy
    ite = IterativeBacktesterWithResults(strategy, symbol=symbol)
    
    # Load the historical data
    ite.load_data(data)
    
    # Run the backtest with optional result printing
    perf, outperf, results = ite.run(print_results=print_results)
    
    return perf, outperf, results
